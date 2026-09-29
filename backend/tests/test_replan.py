"""Перепланирование по правилам организаторов (19.09, 22.09) и инвариант после любого события."""

import pytest

from app.domain.enums import JobStatus, Skill, WorkType
from app.domain.events import ChangeStatus, EventType, ReplanEvent
from app.domain.models import Location, Order, Plan, TimeWindow, minutes_to_time
from app.solver import Solver
from app.solver.replan import ReplanEngine, ReplanError
from tests.helpers import REGIONS, assert_plan_valid, make_engineer, make_order


@pytest.fixture
def day():
    """Две бригады, по две заявки утром и днём у каждой."""
    engineers = [
        make_engineer("e1", "Бригада Первая", [Skill.LOCAL, Skill.CONNECTION, Skill.EMERGENCY]),
        # вторая бригада стартует дальше от заявок: утром на линии только первая
        make_engineer("e2", "Бригада Вторая", [Skill.LOCAL, Skill.CONNECTION], lat=55.76, lon=37.60),
    ]
    orders = [
        make_order("m1", start="10:00", end="12:00", duration=30, lat=55.71, lon=37.69),
        make_order("m2", start="10:00", end="12:00", duration=30, lat=55.73, lon=37.65),
        make_order("d1", start="14:00", end="16:00", duration=70, skill=Skill.CONNECTION, lat=55.72, lon=37.71),
        make_order("d2", start="16:00", end="18:00", duration=30, lat=55.74, lon=37.67),
    ]
    plan = Solver().solve(orders, engineers)
    return orders, engineers, plan


def _where(plan: Plan) -> dict[str, tuple[str, int, JobStatus]]:
    return {j.order_id: (r.engineer_id, j.start_time_min, j.status) for r in plan.routes for j in r.jobs}


def test_started_work_is_frozen_and_not_moved(day) -> None:
    orders, engineers, plan = day
    before = _where(plan)
    event = ReplanEvent(event_type=EventType.ENGINEER_UNAVAILABLE, event_time="10:40", engineer_id="e1")
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    for r in new_plan.routes:
        for j in r.jobs:
            if j.is_frozen and j.status != JobStatus.CANCELLED:
                assert before[j.order_id][:2] == (r.engineer_id, j.start_time_min)
    assert diff.frozen_jobs_count >= 1


def test_new_regular_order_goes_to_free_slot_without_moving_others(day) -> None:
    orders, engineers, plan = day
    new = make_order("n1", start="12:00", end="14:00", duration=30, lat=55.72, lon=37.70)
    event = ReplanEvent(event_type=EventType.NEW_ORDER, event_time="11:00", new_order=new)
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    assert diff.added_order_ids == ["n1"]
    assert not [c for c in diff.changes if c.status == ChangeStatus.REASSIGNED]
    assert "свободный интервал" in diff.summary_ru


def test_emergency_is_placed_as_early_as_possible(day) -> None:
    orders, engineers, plan = day
    emergency = make_order("sos", skill=Skill.EMERGENCY, start="11:00", end="12:00", duration=0)
    event = ReplanEvent(event_type=EventType.URGENT_ORDER, event_time="11:00", new_order=emergency)
    new_plan, _diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    placed = next(o for o in orders2 if o.id == "sos")
    assert placed.kind == WorkType.EMERGENCY and placed.duration_min == 80  # норматив 100 − 20 мин дороги
    assert placed.window.end == "23:59" and placed.released_min == 11 * 60
    route, job = next((r, j) for r in new_plan.routes for j in r.jobs if j.order_id == "sos")
    assert route.engineer_id == "e1"  # только у неё есть навык аварий
    assert job.start_time_min - 11 * 60 <= 120
    assert new_plan.metrics.emergency_within_sla == 1


def test_emergency_evicts_tail_and_reassigns(day) -> None:
    """Авария перестраивает хвост дня; вытесненная заявка уходит другой бригаде или получает причину."""
    orders, engineers, plan = day
    assert {r.engineer_id for r in plan.routes if r.jobs} == {"e1"}
    emergency = make_order("sos", skill=Skill.EMERGENCY, start="12:30", end="23:59", duration=240)
    event = ReplanEvent(event_type=EventType.URGENT_ORDER, event_time="12:30", new_order=emergency)
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    where = _where(new_plan)
    assert where["sos"][0] == "e1" and where["sos"][1] <= 12 * 60 + 30 + 120
    # подключение d1 (14–16) не помещается после аварии у e1 и уходит второй бригаде
    assert where["d1"][0] == "e2"
    assert "e2" in diff.called_in_engineer_ids
    assert any(c.order_id == "d1" and c.status == ChangeStatus.REASSIGNED for c in diff.changes)


def test_cancel_planned_order_frees_the_crew(day) -> None:
    orders, engineers, plan = day
    event = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time="12:00", order_id="d1")
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    assert diff.cancelled_order_ids == ["d1"]
    assert "d1" in new_plan.cancelled_orders and "d1" not in new_plan.unassigned_orders
    assert new_plan.metrics.cancelled_orders == 1
    assert new_plan.metrics.assignment_rate_pct == 100.0  # отмена не считается неназначением


def test_cancel_while_en_route_keeps_the_trip(day) -> None:
    """Отмена, когда бригада уже выехала: доезжает (не разворачиваем) и свободна на месте."""
    orders, engineers, plan = day
    _eid, _start, _ = _where(plan)["d1"]
    job = next(j for r in plan.routes for j in r.jobs if j.order_id == "d1")
    t = job.departure_time_min + 1
    assert t < job.arrival_time_min
    event = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time=minutes_to_time(t), order_id="d1")
    new_plan, _, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    cancelled = next(j for r in new_plan.routes for j in r.jobs if j.order_id == "d1")
    assert cancelled.status == JobStatus.CANCELLED and cancelled.travel_dist_km == job.travel_dist_km
    assert cancelled.end_time_min == job.arrival_time_min
    assert new_plan.metrics.assigned_orders == plan.metrics.assigned_orders - 1


def test_cancel_done_order_is_rejected(day) -> None:
    orders, engineers, plan = day
    event = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time="13:00", order_id="m1")
    with pytest.raises(ReplanError, match="уже выполнена"):
        ReplanEngine().apply_event(plan, event, orders, engineers)


def test_cancel_unknown_order_is_rejected(day) -> None:
    orders, engineers, plan = day
    event = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time="13:00", order_id="missing")
    with pytest.raises(ReplanError, match="не найдена"):
        ReplanEngine().apply_event(plan, event, orders, engineers)


def test_events_must_go_forward_in_time(day) -> None:
    orders, engineers, plan = day
    engine = ReplanEngine()
    e1 = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time="12:00", order_id="d2")
    plan2, _, orders2 = engine.apply_event(plan, e1, orders, engineers)
    e2 = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time="11:00", order_id="d1")
    with pytest.raises(ReplanError, match="раньше предыдущего"):
        engine.apply_event(plan2, e2, orders2, engineers)


def test_unavailable_engineer_gets_no_new_work(day) -> None:
    orders, engineers, plan = day
    engine = ReplanEngine()
    e1 = ReplanEvent(event_type=EventType.ENGINEER_UNAVAILABLE, event_time="12:00", engineer_id="e2")
    plan2, _diff, orders2 = engine.apply_event(plan, e1, orders, engineers)
    route = next(r for r in plan2.routes if r.engineer_id == "e2")
    assert route.unavailable_from_min == 12 * 60
    assert all(j.is_frozen for j in route.jobs)
    new = make_order("n2", start="16:00", end="18:00", duration=30)
    e2 = ReplanEvent(event_type=EventType.NEW_ORDER, event_time="13:00", new_order=new)
    plan3, _, orders3 = engine.apply_event(plan2, e2, orders2, engineers)
    assert_plan_valid(plan3, orders3, engineers)
    assert all(j.order_id != "n2" for j in next(r for r in plan3.routes if r.engineer_id == "e2").jobs)


def _event_for(kind: str, k: int, t: int, plan: Plan, orders: list[Order], engineers) -> ReplanEvent:
    time = minutes_to_time(t)
    ref = orders[(k * 7) % len(orders)]
    if kind in (EventType.NEW_ORDER, EventType.URGENT_ORDER):
        slot = min(((t + 60) // 120 + 1) * 120, 20 * 60)
        new = Order(
            id=f"N{k}",
            skills=[Skill.EMERGENCY] if kind == EventType.URGENT_ORDER else [ref.skills[0]],
            window=TimeWindow(start=minutes_to_time(slot), end=minutes_to_time(slot + 120)),
            duration_min=0,
            location=Location(
                lat=ref.location.lat + 0.004,
                lon=ref.location.lon - 0.003,
                address="новый адрес",
                district=ref.location.district,
            ),
        )
        return ReplanEvent(event_type=kind, event_time=time, new_order=new)
    if kind == EventType.CANCEL_ORDER:
        live = [
            j.order_id
            for r in plan.routes
            for j in r.jobs
            if j.status != JobStatus.CANCELLED and j.end_time_min > t
        ]
        en_route = [
            j.order_id
            for r in plan.routes
            for j in r.jobs
            if j.status != JobStatus.CANCELLED and j.departure_time_min <= t < j.end_time_min
        ]
        pool = en_route if (k % 3 == 1 and en_route) else live
        if k % 5 == 3 and plan.unassigned_orders:
            pool = sorted(plan.unassigned_orders)
        return ReplanEvent(event_type=kind, event_time=time, order_id=pool[0])
    busy = [
        r.engineer_id
        for r in plan.routes
        if r.unavailable_from_min is None and any(not j.is_frozen for j in r.jobs)
    ]
    return ReplanEvent(event_type=kind, event_time=time, engineer_id=busy[k % len(busy)])


@pytest.mark.parametrize("region", REGIONS)
def test_replan_invariants_over_a_day(datasets, region: str) -> None:
    """События каждые 30–60 мин с 10:00 до 20:00, все типы подряд. После каждого события:
    допустимость, каждая заявка ровно в одном состоянии, замороженные работы не сдвинуты,
    бригада в пути не развёрнута, сошедшая бригада не получает работу."""
    orders, engineers = datasets[region]
    plan = Solver().solve(orders, engineers)
    engine = ReplanEngine()
    kinds = [
        EventType.URGENT_ORDER,
        EventType.CANCEL_ORDER,
        EventType.NEW_ORDER,
        EventType.ENGINEER_UNAVAILABLE,
    ]
    t, k, applied = 10 * 60, 0, 0
    while t <= 20 * 60:
        event = _event_for(kinds[k % 4], k, t, plan, orders, engineers)
        before = _where(plan)
        new_plan, diff, orders = engine.apply_event(plan, event, orders, engineers)
        assert_plan_valid(new_plan, orders, engineers)
        for r in new_plan.routes:
            for j in r.jobs:
                if j.is_frozen:
                    was = before.get(j.order_id)
                    assert was is not None and was[0] == r.engineer_id
                    if j.status != JobStatus.CANCELLED:
                        assert was[1] == j.start_time_min
                else:
                    assert j.departure_time_min >= t
                    assert r.unavailable_from_min is None
        assert diff.summary_ru
        plan = new_plan
        applied += 1
        t += 30 + (k % 2) * 30
        k += 1
    assert applied >= 12


def test_manual_assign_moves_order_and_keeps_other_routes(day) -> None:
    orders, engineers, plan = day
    before = _where(plan)
    oid = next(o for o, (eid, _, _) in sorted(before.items()) if eid == "e1")
    event = ReplanEvent(event_type=EventType.MANUAL_ASSIGN, event_time="00:00", order_id=oid, engineer_id="e2")
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    after = _where(new_plan)
    assert after[oid][0] == "e2"
    assert new_plan.as_of_min is None  # утренний план остаётся утренним
    moved = [c for c in diff.changes if c.status == ChangeStatus.REASSIGNED]
    assert [c.order_id for c in moved] == [oid]
    assert "Диспетчер назначил" in diff.summary_ru


def test_manual_assign_rejects_missing_skill(day) -> None:
    orders, engineers, plan = day
    emergency = make_order("sos", skill=Skill.EMERGENCY, start="11:00", end="12:00", duration=0)
    event = ReplanEvent(event_type=EventType.URGENT_ORDER, event_time="10:05", new_order=emergency)
    plan2, _, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    manual = ReplanEvent(event_type=EventType.MANUAL_ASSIGN, event_time="00:00", order_id="sos", engineer_id="e2")
    with pytest.raises(ReplanError, match="нет навыка"):
        ReplanEngine().apply_event(plan2, manual, orders2, engineers)


def test_manual_assign_rejects_started_work(day) -> None:
    orders, engineers, plan = day
    first = next(r.jobs[0] for r in plan.routes if r.jobs)
    t = minutes_to_time(first.start_time_min + 5)
    event = ReplanEvent(event_type=EventType.CANCEL_ORDER, event_time=t, order_id=next(
        j.order_id for r in plan.routes for j in r.jobs if j.start_time_min > first.start_time_min + 60
    ))
    plan2, _, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    owner = next(r.engineer_id for r in plan2.routes if any(j.order_id == first.order_id for j in r.jobs))
    other = "e2" if owner == "e1" else "e1"
    manual = ReplanEvent(
        event_type=EventType.MANUAL_ASSIGN, event_time="00:00", order_id=first.order_id, engineer_id=other
    )
    with pytest.raises(ReplanError, match="нельзя переназначить"):
        ReplanEngine().apply_event(plan2, manual, orders2, engineers)


def test_new_regular_order_does_not_call_in_a_reserve_crew() -> None:
    """Эксперты 29.09: дополнительную бригаду выводят при форс-мажоре (авария, сход), не ради
    обычной заявки. Если у бригад на линии нет места, заявка остаётся без исполнителя с причиной."""
    engineers = [
        make_engineer("e1", "Бригада Первая", [Skill.LOCAL]),
        make_engineer("e2", "Бригада Резерв", [Skill.LOCAL]),
    ]
    orders = [make_order("a", start="10:00", end="12:00", duration=150)]
    plan = Solver().solve(orders, engineers)
    assert {r.engineer_id for r in plan.routes if r.jobs} == {"e1"}
    new = make_order("n", start="10:00", end="11:30", duration=60)
    event = ReplanEvent(event_type=EventType.NEW_ORDER, event_time="10:05", new_order=new)
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, orders, engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    assert "n" in new_plan.unassigned_orders and not diff.called_in_engineer_ids
    assert "не на линии" in new_plan.unassigned_orders["n"]
    # авария — форс-мажор: резерв вызывается
    sos = make_order("sos", skill=Skill.LOCAL, start="10:10", end="23:59", duration=60)
    sos = sos.model_copy(update={"work_type": WorkType.EMERGENCY})
    urgent = ReplanEvent(event_type=EventType.URGENT_ORDER, event_time="10:10", new_order=sos)
    engineers2 = [e.model_copy(update={"skills": [Skill.LOCAL, Skill.EMERGENCY]}) for e in engineers]
    plan3, diff3, orders3 = ReplanEngine().apply_event(new_plan, urgent, orders2, engineers2)
    assert_plan_valid(plan3, orders3, engineers2)
    assert diff3.called_in_engineer_ids == ["e2"]


def test_new_emergency_never_drops_another_emergency() -> None:
    """Новая авария может снять обычные заявки, но не другую аварию: если иначе не поставить,
    она остаётся без исполнителя с понятной причиной, а прежняя авария — в плане."""
    engineers = [
        make_engineer("e1", "Бригада Аварийная", [Skill.LOCAL, Skill.EMERGENCY], shift_end="13:00"),
    ]
    first = make_order("a1", skill=Skill.EMERGENCY, start="11:00", end="13:00", duration=80)
    first = first.model_copy(update={"work_type": WorkType.EMERGENCY})
    plan = Solver().solve([first], engineers)
    assert plan.routes[0].jobs[0].order_id == "a1"
    new = make_order("a2", skill=Skill.EMERGENCY, start="10:30", end="23:59", duration=0)
    event = ReplanEvent(event_type=EventType.URGENT_ORDER, event_time="10:30", new_order=new)
    new_plan, diff, orders2 = ReplanEngine().apply_event(plan, event, [first], engineers)
    assert_plan_valid(new_plan, orders2, engineers)
    assert [j.order_id for j in new_plan.routes[0].jobs] == ["a1"]
    assert "a2" in new_plan.unassigned_orders
    assert "заняты другими авариями" in diff.summary_ru
