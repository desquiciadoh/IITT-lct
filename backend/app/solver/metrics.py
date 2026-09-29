from typing import Any

from app.domain.enums import JobStatus, Transport
from app.domain.models import Engineer, Order, Plan, PlanMetrics, Route
from app.solver.objective import SLA_REACTION_MIN, release_min


def calculate_plan_metrics(
    orders: list[Order],
    engineers: list[Engineer],
    routes: list[Route],
    unassigned_orders: dict[str, Any],
    cancelled_orders: dict[str, Any] | None = None,
    *,
    day_start_min: int | None = None,
    extra_crews_needed: int = 0,
) -> PlanMetrics:
    """Метрики ТЗ §2.3 (бригады, пробег) плюс выполнение, отмены и реакция на аварии."""
    cancelled_orders = cancelled_orders or {}
    orders_map = {o.id: o for o in orders}
    eng_map = {e.id: e for e in engineers}
    if day_start_min is None:
        day_start_min = min((e.shift.start_min for e in engineers), default=600)

    total_orders = len(orders)
    assigned_orders = sum(1 for r in routes for j in r.jobs if j.status != JobStatus.CANCELLED)
    active_base = total_orders - len(cancelled_orders)
    assignment_rate_pct = (
        round((assigned_orders / active_base) * 100.0, 1) if active_base > 0 else 0.0
    )

    reactions: list[int] = []
    for r in routes:
        for j in r.jobs:
            order = orders_map.get(j.order_id)
            if order is None or not order.is_emergency or j.status == JobStatus.CANCELLED:
                continue
            reactions.append(max(0, j.start_time_min - release_min(order, day_start_min)))

    car_km = sum(
        r.total_distance_km
        for r in routes
        if r.engineer_id in eng_map and eng_map[r.engineer_id].transport == Transport.CAR
    )

    # загрузка = (дорога + работа) / смена по бригадам на линии (организаторы: < 50 % — лишний человек)
    busy_min = shift_min = low_load = 0
    for r in routes:
        eng = eng_map.get(r.engineer_id)
        if eng is None or not any(j.status != JobStatus.CANCELLED for j in r.jobs):
            continue
        busy = r.total_work_time_min + r.total_travel_time_min
        length = max(1, eng.shift.end_min - eng.shift.start_min)
        busy_min += busy
        shift_min += length
        low_load += 1 if busy < 0.5 * length else 0

    return PlanMetrics(
        total_orders=total_orders,
        assigned_orders=assigned_orders,
        unassigned_orders=len(unassigned_orders),
        cancelled_orders=len(cancelled_orders),
        assignment_rate_pct=assignment_rate_pct,
        active_engineers_count=sum(
            1 for r in routes if any(j.status != JobStatus.CANCELLED for j in r.jobs)
        ),
        total_engineers_count=len(engineers),
        total_distance_km=round(sum(r.total_distance_km for r in routes), 2),
        car_distance_km=round(car_km, 2),
        total_travel_time_min=sum(r.total_travel_time_min for r in routes),
        total_work_time_min=sum(r.total_work_time_min for r in routes),
        emergency_orders=sum(1 for o in orders if o.is_emergency and o.id not in cancelled_orders),
        emergency_within_sla=sum(1 for x in reactions if x <= SLA_REACTION_MIN),
        emergency_avg_reaction_min=round(sum(reactions) / len(reactions), 1) if reactions else None,
        emergency_max_reaction_min=max(reactions) if reactions else None,
        extra_crews_needed=extra_crews_needed,
        avg_load_pct=round(100.0 * busy_min / shift_min, 1) if shift_min else None,
        low_load_crews=low_load,
        engineer_distances={r.engineer_id: r.total_distance_km for r in routes},
        engineer_order_counts={
            r.engineer_id: sum(1 for j in r.jobs if j.status != JobStatus.CANCELLED) for r in routes
        },
    )


def compare_plans(optimized_plan: Plan, baseline_plan: Plan) -> dict[str, Any]:
    """Сравнивает оптимизированный план с базовым вариантом FIFO (ТЗ §2.3) и вычисляет выигрыш."""
    opt_m = optimized_plan.metrics
    base_m = baseline_plan.metrics

    def pct_saving(base: float, opt: float) -> float:
        return round(((base - opt) / base) * 100.0, 1) if base > 0 else 0.0

    def km_per_order(m: PlanMetrics) -> float:
        return round(m.total_distance_km / m.assigned_orders, 2) if m.assigned_orders > 0 else 0.0

    def block(m: PlanMetrics) -> dict[str, Any]:
        return {
            "assigned_orders": m.assigned_orders,
            "total_orders": m.total_orders,
            "assignment_rate_pct": m.assignment_rate_pct,
            "active_crews": m.active_engineers_count,
            "total_crews": m.total_engineers_count,
            "total_distance_km": m.total_distance_km,
            "km_per_order": km_per_order(m),
            "total_travel_time_min": m.total_travel_time_min,
            "emergency_avg_reaction_min": m.emergency_avg_reaction_min,
            "emergency_within_sla": m.emergency_within_sla,
            "emergency_orders": m.emergency_orders,
            "avg_load_pct": m.avg_load_pct,
            "low_load_crews": m.low_load_crews,
        }

    return {
        "optimized": block(opt_m),
        "baseline": block(base_m),
        "delta": {
            "crew_count": opt_m.active_engineers_count - base_m.active_engineers_count,
            "crew_savings_pct": pct_saving(base_m.active_engineers_count, opt_m.active_engineers_count),
            "total_distance_km": round(opt_m.total_distance_km - base_m.total_distance_km, 2),
            "distance_savings_pct": pct_saving(base_m.total_distance_km, opt_m.total_distance_km),
            "km_per_order": round(km_per_order(opt_m) - km_per_order(base_m), 2),
            "assigned_orders": opt_m.assigned_orders - base_m.assigned_orders,
            "travel_time_min": opt_m.total_travel_time_min - base_m.total_travel_time_min,
        },
    }
