import os
import subprocess
import sys

import pytest

from app.domain.enums import ReasonCode, Skill, Transport
from app.domain.models import Location, TimeWindow
from app.solver import BaselineSolver, Solver
from app.solver.explain import ExplanationGenerator
from tests.helpers import REGIONS, assert_plan_valid, make_engineer, make_order


def test_time_window_ignores_tampered_derived_minutes() -> None:
    window = TimeWindow(start="10:00", end="12:00", start_min=-1, end_min=9999)
    assert (window.start_min, window.end_min) == (600, 720)


@pytest.mark.parametrize("value", ["24:00", "10:60", "-1:00", "10:000"])
def test_time_window_rejects_invalid_time(value: str) -> None:
    with pytest.raises(ValueError):
        TimeWindow(start=value, end="12:00")


def test_location_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        Location(lat=float("nan"), lon=37.0, address="x", district="x")


@pytest.mark.parametrize("region", REGIONS)
def test_plans_satisfy_all_constraints(datasets, region: str) -> None:
    orders, engineers = datasets[region]
    assert_plan_valid(Solver().solve(orders, engineers), orders, engineers)
    assert_plan_valid(BaselineSolver().solve(orders, engineers), orders, engineers)


@pytest.mark.parametrize("region", REGIONS)
def test_optimized_beats_fifo_baseline(datasets, region: str) -> None:
    """ТЗ §2.3: сравнение с базовым вариантом по числу бригад и пробегу."""
    orders, engineers = datasets[region]
    opt = Solver().solve(orders, engineers).metrics
    base = BaselineSolver().solve(orders, engineers).metrics
    assert opt.assigned_orders > base.assigned_orders
    assert opt.active_engineers_count <= base.active_engineers_count
    assert opt.total_distance_km / opt.assigned_orders < base.total_distance_km / base.assigned_orders


def test_solver_determinism(datasets) -> None:
    orders, engineers = datasets["east"]
    a = Solver().solve(orders, engineers)
    b = Solver().solve(orders, engineers)
    assert a.model_dump() == b.model_dump()


def test_solver_cross_process_determinism() -> None:
    """Одинаковый результат в разных процессах при разном PYTHONHASHSEED."""
    cmd = [sys.executable, "-m", "app.cli", "solve", "--region", "southeast"]
    outputs = []
    for seed in ("0", "99999"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONIOENCODING"] = "utf-8"
        res = subprocess.run(cmd, env=env, capture_output=True, check=False, encoding="utf-8")
        assert res.returncode == 0, res.stderr
        assert res.stdout
        outputs.append(res.stdout)
    assert outputs[0] == outputs[1]


def test_unassigned_reasons_are_specific() -> None:
    """Причина — реальный код по всем бригадам, а не отказы первых бригад по списку."""
    engineers = [
        make_engineer("walker", skills=[Skill.CONNECTION], transport=Transport.FOOT),
        make_engineer("driver", skills=[Skill.LOCAL], transport=Transport.CAR),
        make_engineer("busy", skills=[Skill.LOCAL], transport=Transport.CAR),
    ]
    orders = [
        make_order("no-skill", skill=Skill.EMERGENCY),
        make_order("no-transport", skill=Skill.LOCAL, transport=Transport.FOOT),
        make_order("late", skill=Skill.LOCAL, start="20:00", end="22:00"),
        # три одинаковые заявки в одном окне на две бригады с навыком: одна останется
        make_order("b1", skill=Skill.LOCAL, start="10:00", end="10:30", duration=120),
        make_order("b2", skill=Skill.LOCAL, start="10:00", end="10:30", duration=120),
        make_order("b3", skill=Skill.LOCAL, start="10:00", end="10:30", duration=120),
    ]
    plan = Solver().solve(orders, engineers)
    details = plan.unassigned_details
    assert details["no-skill"].code == ReasonCode.SKILL
    assert details["no-transport"].code == ReasonCode.TRANSPORT
    assert details["late"].code == ReasonCode.SHIFT_WINDOW
    busy = [oid for oid in ("b1", "b2", "b3") if oid in details]
    assert len(busy) == 1 and details[busy[0]].code == ReasonCode.BUSY
    assert "Обе подходящие бригады заняты" in details[busy[0]].text
    assert plan.metrics.extra_crews_needed >= 1


def test_emergency_priority_when_capacity_is_short() -> None:
    """При нехватке ресурсов: авария → подключение → ремонт (разъяснение 19.09)."""
    eng = make_engineer(skills=[Skill.LOCAL, Skill.CONNECTION, Skill.EMERGENCY])
    orders = [
        make_order("local", skill=Skill.LOCAL, start="10:00", end="10:30", duration=100),
        make_order("conn", skill=Skill.CONNECTION, start="10:00", end="10:30", duration=100),
        make_order("emerg", skill=Skill.EMERGENCY, start="10:00", end="10:30", duration=100),
    ]
    plan = Solver().solve(orders, [eng])
    assert [j.order_id for j in plan.routes[0].jobs] == ["emerg"]


def test_explanations_are_short_and_complete(datasets) -> None:
    orders, engineers = datasets["east"]
    plan = Solver().solve(orders, engineers)
    texts = ExplanationGenerator.enrich_plan_explanations(plan, orders, engineers)
    assert set(texts) == {o.id for o in orders}
    job = plan.routes[0].jobs[0] if plan.routes[0].jobs else next(j for r in plan.routes for j in r.jobs)
    text = texts[job.order_id]
    for word in ("Квалификация", "Транспорт", "Время", "Маршрут"):
        assert word in text
    assert "eng_" not in text and "connection" not in text
    assert len(text.splitlines()) <= 8
    for oid in plan.unassigned_orders:
        assert "Причина:" in texts[oid]
    routes = ExplanationGenerator.route_explanations(plan, orders, engineers)
    assert set(routes) == {e.id for e in engineers}
