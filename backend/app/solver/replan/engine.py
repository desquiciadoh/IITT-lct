"""Перепланирование в течение дня по правилам организаторов (19.09, 22.09).

- Начатая работа не прерывается; к заявке, куда бригада уже выехала, она доезжает (допущение).
- Новая обычная заявка встаёт только в свободный интервал бригад на линии: чужие заявки не
  переносятся к другим бригадам, их время может сдвинуться в пределах окон (со штрафом за сдвиг).
  Резервную бригаду вызываем только при форс-мажоре — аварии или сходе бригады (29.09).
- Авария ставится как можно раньше (ориентир — 2 часа от поступления) и может перестроить хвост
  дня бригады; вытесненные заявки переназначаются, иначе помечаются «не назначена» с причиной.
- Отмена освобождает время бригады; в освободившийся интервал пробуем поставить неназначенные.
- Сход бригады: её невыполненные заявки распределяются по остальным.
- Ручное назначение: диспетчер переносит заявку к выбранной бригаде; проверка — тем же оценщиком,
  остальные маршруты не перестраиваются, время события — момент последнего разреза плана.
Любая заявка после события ровно в одном состоянии: назначена, не назначена с причиной или отменена.
"""

from dataclasses import dataclass

from app.domain.enums import JobStatus, Priority, Skill, WorkType
from app.domain.events import EventType, PlanDiff, ReplanEvent
from app.domain.models import Engineer, Order, Plan, TimeWindow, minutes_to_time, time_to_minutes
from app.geo.base import DistanceProvider
from app.geo.matrix import as_matrix
from app.solver.diagnose import crew_name, diagnose, estimate_extra_crews, km_ru
from app.solver.evaluator import RouteEvaluator
from app.solver.fleet import Fleet
from app.solver.objective import (
    CREW_PENALTY,
    LATE_KM_PER_MIN,
    STABILITY_KM_PER_MIN,
)
from app.solver.plan_builder import build_plan
from app.solver.replan.alternatives import check, locked_reason, reason_text
from app.solver.replan.diff import compute_diff
from app.solver.replan.state import DayState, fleet_from_state, state_at
from app.solver.search.insertion import insert_pool

DAY_END_MIN = 23 * 60 + 59
EVICTION_KM = 5.0  # перенос чужой заявки ради аварии «стоит» 5 км


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


class ReplanError(ValueError):
    """Событие нельзя применить (понятный диспетчеру текст)."""


@dataclass
class _Outcome:
    summary: str
    lost_context: dict[str, str]  # order_id -> контекст причины («вытеснена аварией #X»)


def normalize_new_order(order: Order, event_type: EventType, t: int) -> Order:
    """Заявка, поступившая днём: норматив по виду работ, окно не раньше момента поступления.

    Для аварии окно — до конца дня, а начать нужно как можно раньше (19.09, п. 5).
    """
    is_emergency = event_type == EventType.URGENT_ORDER or order.is_emergency
    kind = WorkType.EMERGENCY if is_emergency else order.kind
    duration = order.duration_min if order.duration_min > 0 else kind.work_min
    if is_emergency:
        start = max(order.window.start_min, t)
        window = TimeWindow(start=minutes_to_time(start), end=minutes_to_time(DAY_END_MIN))
        return order.model_copy(
            update={
                "skills": [Skill.EMERGENCY],
                "priority": Priority.URGENT,
                "work_type": WorkType.EMERGENCY,
                "window": window,
                "duration_min": duration,
                "released_min": t,
            }
        )
    return order.model_copy(
        update={"work_type": kind, "duration_min": duration, "released_min": t}
    )


class ReplanEngine:
    """Движок динамического перепланирования при событиях в течение дня (ТЗ §2.1.6, §2.4)."""

    def __init__(
        self,
        distance_provider: DistanceProvider | None = None,
        crew_penalty: float = CREW_PENALTY,
    ):
        self.matrix = as_matrix(distance_provider)
        self.distance_provider = self.matrix
        self.crew_penalty = crew_penalty

    # ------------------------------------------------------------------ публичный API

    def apply_event(
        self,
        current_plan: Plan,
        event: ReplanEvent,
        orders: list[Order],
        engineers: list[Engineer],
    ) -> tuple[Plan, PlanDiff, list[Order]]:
        """Применяет событие. Возвращает новый план, дифф и актуальный список заявок."""
        manual = event.event_type == EventType.MANUAL_ASSIGN
        if manual:
            # ручное назначение не сдвигает «сейчас»: разрез — на момент последнего события
            t = current_plan.as_of_min or 0
            event = event.model_copy(update={"event_time": minutes_to_time(t)})
        else:
            try:
                t = time_to_minutes(event.event_time)
            except ValueError as exc:
                raise ReplanError(
                    f"Время события «{event.event_time}» должно быть в формате ЧЧ:ММ"
                ) from exc
        if not 0 <= t <= DAY_END_MIN:
            raise ReplanError(f"Время события «{event.event_time}» вне суток")
        if current_plan.as_of_min is not None and t < current_plan.as_of_min:
            raise ReplanError(
                f"Событие в {event.event_time} раньше предыдущего "
                f"({minutes_to_time(current_plan.as_of_min)}): события применяются по времени."
            )

        all_orders = list(orders)
        if event.event_type in (EventType.NEW_ORDER, EventType.URGENT_ORDER):
            if event.new_order is None:
                raise ReplanError("Для новой заявки нужно передать её параметры (new_order)")
            if any(o.id == event.new_order.id for o in orders):
                raise ReplanError(f"Заявка #{event.new_order.id} уже есть в плане")
            new_order = normalize_new_order(event.new_order, event.event_type, t)
            event = event.model_copy(update={"new_order": new_order})
            all_orders.append(new_order)

        orders_map = {o.id: o for o in all_orders}
        ev = RouteEvaluator(
            self.matrix, day_start_min=min((e.shift.start_min for e in engineers), default=600)
        )
        ds = state_at(current_plan, t, orders_map, engineers)
        cancelled = dict(current_plan.cancelled_orders)
        unassigned_before = [oid for oid in current_plan.unassigned_orders if oid not in cancelled]

        handlers = {
            EventType.NEW_ORDER: self._new_order,
            EventType.URGENT_ORDER: self._new_order,
            EventType.CANCEL_ORDER: self._cancel,
            EventType.ENGINEER_UNAVAILABLE: self._unavailable,
            EventType.MANUAL_ASSIGN: self._manual,
        }
        handler = handlers.get(event.event_type)
        if handler is None:
            raise ReplanError(f"Неизвестный тип события: {event.event_type}")
        # обработчик может изменить ds (заморозка, отмена) до построения маршрутов
        fleet, pool, outcome = handler(ev, engineers, ds, event, orders_map, cancelled, unassigned_before)

        assigned = fleet.assigned_ids() | {
            j.order_id for p in ds.prefixes.values() for j in p if j.status != JobStatus.CANCELLED
        }
        left_ids = [o.id for o in pool if o.id not in assigned]
        left_ids += [
            oid for oid in unassigned_before if oid not in assigned and oid not in left_ids
        ]
        left_ids = [oid for oid in left_ids if oid not in cancelled]
        unassigned = {}
        for oid in left_ids:
            info = diagnose(
                fleet, orders_map[oid], event_time_min=t, context=outcome.lost_context.get(oid)
            )
            old = current_plan.unassigned_details.get(oid)
            if old is not None and oid not in outcome.lost_context and oid in unassigned_before:
                info = old  # причина по заявке, не затронутой событием, остаётся прежней
            unassigned[oid] = info

        extra = estimate_extra_crews(ev, engineers, [orders_map[oid] for oid in left_ids])
        new_plan = build_plan(
            fleet,
            all_orders,
            unassigned,
            prefixes=ds.prefixes,
            cancelled=cancelled,
            unavailable_from=ds.unavailable_from,
            as_of_min=current_plan.as_of_min if manual else t,
            extra_crews_needed=extra,
        )
        diff = compute_diff(
            current_plan,
            new_plan,
            event,
            engineers,
            frozen_count=ds.frozen_count(),
            summary=outcome.summary,
        )
        return new_plan, diff, all_orders

    # ------------------------------------------------------------------ события

    def _new_order(self, ev, engineers, ds: DayState, event, orders_map, cancelled, unassigned_before):
        order = event.new_order
        t = ds.t
        fleet, pool = fleet_from_state(ev, engineers, ds)
        lost_context: dict[str, str] = {}
        names = {e.id: crew_name(e.name) for e in engineers}

        if order.window.end_min < t:
            return fleet, pool + [order], _Outcome(
                f"Новая заявка #{order.id}: окно {order.window.start}–{order.window.end} уже прошло.",
                lost_context,
            )

        if order.is_emergency:
            placed_eid, evicted, left, blocked = self._place_emergency(fleet, order, t)
            for o in left:
                lost_context[o.id] = f"Вытеснена аварией #{order.id}."
            pool = pool + left
            if placed_eid is None:
                pool.append(order)
                if blocked:
                    lost_context[order.id] = (
                        "Бригады с навыком «Аварийные работы» до конца смены заняты другими "
                        "авариями, а другие аварии не снимаем."
                    )
                    summary = (
                        f"Авария #{order.id} ({minutes_to_time(t)}): бригады с навыком «Аварийные "
                        "работы» до конца смены заняты другими авариями. Назначьте вручную или "
                        "вызовите бригаду."
                    )
                else:
                    summary = (
                        f"Авария #{order.id} ({minutes_to_time(t)}): ни одна бригада с навыком "
                        f"«Аварийные работы» не может её взять."
                    )
            else:
                st = fleet.states[placed_eid]
                start = st.starts[st.order_ids.index(order.id)]
                reaction = start - t
                sla = "в пределах 2 ч" if reaction <= 120 else "сверх ориентира 2 ч"
                summary = (
                    f"Авария #{order.id}: {names[placed_eid]} начнёт в {minutes_to_time(start)} — "
                    f"через {reaction} мин после поступления ({sla})."
                )
                if evicted:
                    moved = len(evicted) - len(left)
                    summary += f" Хвост дня перестроен: перенесено {moved}"
                    summary += f", снято {len(left)}." if left else "."
            return fleet, pool, _Outcome(summary, lost_context)

        # обычная заявка — не форс-мажор: резервную бригаду ради неё не вызываем (29.09),
        # решение о вызове остаётся за диспетчером (ручное назначение)
        left = insert_pool(
            fleet,
            [order],
            crew_penalty=self.crew_penalty,
            allow_activation=False,
            stability=STABILITY_KM_PER_MIN,
        )
        if left:
            summary = (
                f"Новая заявка #{order.id} ({order.kind.label_ru.lower()}): у бригад на линии "
                "нет свободного интервала."
            )
            return fleet, pool + left, _Outcome(summary, lost_context)
        eid = fleet.owner_of(order.id)
        st = fleet.states[eid]
        start = st.starts[st.order_ids.index(order.id)]
        summary = (
            f"Новая заявка #{order.id} ({order.kind.label_ru.lower()}) поставлена в свободный "
            f"интервал: {names[eid]}, начало в {minutes_to_time(start)}. Другие заявки не переносились."
        )
        return fleet, pool, _Outcome(summary, lost_context)

    def _cancel(self, ev, engineers, ds: DayState, event, orders_map, cancelled, unassigned_before):
        oid = event.order_id
        t = ds.t
        if not oid or oid not in orders_map:
            raise ReplanError(f"Заявка #{oid} не найдена")
        if oid in cancelled:
            raise ReplanError(f"Заявка #{oid} уже отменена")
        {e.id: crew_name(e.name) for e in engineers}
        when = minutes_to_time(t)
        freed_eid: str | None = None

        for eid, prefix in ds.prefixes.items():
            for i, job in enumerate(prefix):
                if job.order_id != oid:
                    continue
                if job.status == JobStatus.DONE:
                    raise ReplanError(f"Заявка #{oid} уже выполнена к {when} — отменять нечего")
                if job.status == JobStatus.IN_PROGRESS:
                    update = {"end_time_min": t, "status": JobStatus.CANCELLED}
                    note = f"Отменена клиентом в {when} во время работ; бригада свободна сразу."
                else:  # EN_ROUTE: бригаду в пути не разворачиваем — доезжает и свободна на месте
                    update = {
                        "start_time_min": job.arrival_time_min,
                        "end_time_min": job.arrival_time_min,
                        "status": JobStatus.CANCELLED,
                    }
                    note = (
                        f"Отменена клиентом в {when}, когда бригада была в пути; "
                        f"свободна на месте с {minutes_to_time(job.arrival_time_min)}."
                    )
                prefix[i] = job.model_copy(update=update)
                cancelled[oid] = note
                freed_eid = eid
                break
            if freed_eid:
                break

        if freed_eid is None:
            for eid, tail in ds.tails.items():
                if any(o.id == oid for o in tail):
                    ds.tails[eid] = [o for o in tail if o.id != oid]
                    freed_eid = eid
                    break
            if freed_eid is None and oid not in unassigned_before:
                raise ReplanError(
                    f"Заявка #{oid} не находится в текущем плане и не может быть отменена"
                )
            cancelled[oid] = f"Отменена клиентом в {when}."

        eng_map = {e.id: e for e in engineers}
        ds.starts = {e.id: ds.start_after(e, orders_map) for e in engineers}
        fleet, pool = fleet_from_state(ev, engineers, ds)

        # в освободившийся интервал — ранее неназначенные заявки (только свободные окна, без новых бригад)
        candidates = [
            orders_map[x]
            for x in unassigned_before
            if x != oid and orders_map[x].window.end_min >= t
        ]
        before = fleet.assigned_ids()
        insert_pool(
            fleet,
            candidates,
            crew_penalty=self.crew_penalty,
            allow_activation=False,
            stability=STABILITY_KM_PER_MIN,
        )
        filled = sorted(fleet.assigned_ids() - before)

        if freed_eid is None:
            summary = f"Заявка #{oid} отменена клиентом в {when}; она не была назначена."
        else:
            summary = f"Заявка #{oid} отменена клиентом в {when}; {crew_name(eng_map[freed_eid].name)} освобождена."
        if filled:
            summary += " В освободившееся время поставлены: " + ", ".join(f"#{x}" for x in filled) + "."
        return fleet, pool, _Outcome(summary, {})

    def _unavailable(self, ev, engineers, ds: DayState, event, orders_map, cancelled, unassigned_before):
        eid = event.engineer_id
        eng_map = {e.id: e for e in engineers}
        if not eid or eid not in eng_map:
            raise ReplanError(f"Бригада {eid} не найдена")
        if not ds.available.get(eid, True):
            raise ReplanError(f"{crew_name(eng_map[eid].name)} уже сошла с линии")
        t = ds.t
        when = minutes_to_time(t)

        # выполненные и начатые работы остаются за бригадой; к кому ещё не приехала — в пул
        prefix = ds.prefixes.get(eid, [])
        keep = [j for j in prefix if j.status != JobStatus.EN_ROUTE]
        orphan_ids = [j.order_id for j in prefix if j.status == JobStatus.EN_ROUTE]
        ds.prefixes[eid] = keep
        orphans = [orders_map[x] for x in orphan_ids] + list(ds.tails.get(eid, []))
        ds.tails[eid] = []
        ds.available[eid] = False
        ds.unavailable_from[eid] = t
        ds.starts = {e.id: ds.start_after(e, orders_map) for e in engineers}

        fleet, pool = fleet_from_state(ev, engineers, ds)
        context = f"{crew_name(eng_map[eid].name)} сошла с линии в {when}."
        lost_context = {o.id: context for o in orphans}

        emergencies = [o for o in orphans if o.is_emergency]
        others = [o for o in orphans if not o.is_emergency]
        left: list[Order] = []
        for em in sorted(emergencies, key=lambda o: (o.window.start_min, o.id)):
            placed, _, evicted_left, _ = self._place_emergency(fleet, em, t)
            for o in evicted_left:
                lost_context[o.id] = f"Вытеснена аварией #{em.id}."
            left += evicted_left
            if placed is None:
                left.append(em)
        left += insert_pool(
            fleet, others, crew_penalty=self.crew_penalty, stability=STABILITY_KM_PER_MIN
        )
        moved = len(orphans) - sum(1 for o in left if o.id in {x.id for x in orphans})
        summary = (
            f"{crew_name(eng_map[eid].name)} сошла с линии в {when}. Невыполненных заявок: {len(orphans)}; "
            f"перераспределено {moved}, без исполнителя {len(orphans) - moved}."
        )
        return fleet, pool + left, _Outcome(summary, lost_context)

    def _manual(self, ev, engineers, ds: DayState, event, orders_map, cancelled, unassigned_before):
        oid, eid = event.order_id, event.engineer_id
        eng_map = {e.id: e for e in engineers}
        if not oid or oid not in orders_map:
            raise ReplanError(f"Заявка #{oid} не найдена")
        if not eid or eid not in eng_map:
            raise ReplanError(f"Бригада {eid} не найдена")
        locked = locked_reason(ds, cancelled, oid)
        if locked:
            raise ReplanError(f"Заявку #{oid} нельзя переназначить: {_lower_first(locked)}")
        eng, order = eng_map[eid], orders_map[oid]
        owner: str | None = None
        for e_id, tail in ds.tails.items():
            if any(o.id == oid for o in tail):
                if e_id == eid:
                    raise ReplanError(f"Заявка #{oid} уже у бригады {eng.name}")
                ds.tails[e_id] = [o for o in tail if o.id != oid]
                owner = e_id
                break
        fleet, pool = fleet_from_state(ev, engineers, ds)
        code, ins = check(fleet, eng, order)
        if code is not None or ins is None:
            raise ReplanError(
                f"{crew_name(eng.name)} не может взять заявку #{oid}: "
                f"{_lower_first(reason_text(code, order, eng))}"
            )
        fleet.states[eid] = ev.insert(fleet.states[eid], order, ins.pos)
        was = f"было: {crew_name(eng_map[owner].name)}" if owner else "была без исполнителя"
        summary = (
            f"Диспетчер назначил заявку #{oid} ({order.kind.label_ru.lower()}) — "
            f"{crew_name(eng.name)}, начало в {minutes_to_time(ins.start_min)}, "
            f"+{km_ru(ins.delta_km)} к маршруту ({was}). Другие маршруты не менялись."
        )
        return fleet, pool, _Outcome(summary, {})

    # ------------------------------------------------------------------ авария

    def _place_emergency(
        self, fleet: Fleet, order: Order, t: int
    ) -> tuple[str | None, list[Order], list[Order], bool]:
        """Ставит аварию как можно раньше, при необходимости вытесняя хвост маршрута.

        Перебирает бригаду и позицию; заявки после аварии, которые перестают успевать, снимаются
        и переназначаются regret-вставкой. Ради аварии можно снять обычные заявки (время с клиентом
        согласует поддержка, 19.09), но не другие аварии: если без этого её не поставить, она
        остаётся без исполнителя. Выбор: (1) меньше аварий позже 2 ч — новая и уже стоящие в плане;
        (2) меньше снятых заявок с учётом приоритета; (3) меньше бригад: резерв выводится, только
        если иначе авария опоздает или заявки останутся без исполнителя; (4) взвешенно: км, время
        реакции и число переносов.
        Возвращает (бригада | None, вытесненные, из них не поставленные никуда, мешают ли другие
        аварии — True, если поставить можно было только сняв другую аварию).
        """
        ev = fleet.evaluator
        best: tuple[tuple, dict, list[Order], list[Order], str] | None = None
        base = fleet.snapshot()
        blocked = False

        for eng in fleet.engineers:
            eid = eng.id
            if not fleet.available.get(eid, True) or ev.pair_reason(eng, order) is not None:
                continue
            st = base[eid]
            tail = st.orders
            for p in range(len(tail) + 1):
                head_state = ev.state(eng, tail[:p] + (order,), st.start)
                if head_state is None:
                    continue
                em_start = head_state.starts[-1]
                kept = list(tail[:p]) + [order]
                evicted: list[Order] = []
                for o in tail[p:]:
                    if ev.state(eng, kept + [o], st.start) is None:
                        evicted.append(o)
                    else:
                        kept.append(o)
                fleet.restore(base)
                fleet.states[eid] = ev.state(eng, kept, st.start)
                # вытесненные не возвращаем в этот маршрут: иначе они встанут перед аварией
                left = insert_pool(
                    fleet,
                    evicted,
                    crew_penalty=self.crew_penalty,
                    stability=STABILITY_KM_PER_MIN,
                    exclude={eid},
                ) if evicted else []
                if any(o.is_emergency for o in left):
                    blocked = True  # другую аварию не снимаем
                    continue
                final = fleet.states[eid]
                em_start = final.starts[final.order_ids.index(order.id)]
                key = (
                    fleet.breaches(),
                    sum(o.kind.weight for o in left),
                    fleet.crews(),
                    round(
                        fleet.cost()
                        + LATE_KM_PER_MIN * (em_start - t)
                        + EVICTION_KM * len(evicted),
                        4,
                    ),
                    eid,
                    p,
                )
                if best is None or key < best[0]:
                    best = (key, fleet.snapshot(), evicted, left, eid)

        if best is None:
            fleet.restore(base)
            return None, [], [], blocked
        _, snap, evicted, left, eid = best
        fleet.restore(snap)
        return eid, evicted, left, False
