"""Локальный поиск по маршрутам Fleet: relocate, swap, 2-opt, or-opt.

Каждый ход оценивается только по двум изменённым маршрутам (вставка — за O(1) через RouteEvaluator),
а принимается, если строго улучшает (аварии позже SLA, число бригад, стоимость маршрутов) — в этом
порядке. Новые бригады не выводятся. Ограничение — число проходов, а не время: результат
детерминирован на любой машине. Дорогие ходы (or-opt, swap) сначала проверяются точной нижней
границей прироста стоимости, и маршрут пересчитывается, только если ход может улучшить план.
"""

from collections.abc import Callable

from app.domain.models import Order
from app.solver import objective as obj
from app.solver.evaluator import RouteState
from app.solver.fleet import Fleet

EPS = 1e-6

Delta = tuple[int, int, float]  # (Δаварий позже SLA, Δбригад, Δстоимости)
LegFn = Callable[[int, int], tuple[float, int]]


def _improves(delta: Delta) -> bool:
    breaches, crews, cost = delta
    if breaches != 0:
        return breaches < 0
    if crews != 0:
        return crews < 0
    return cost < -EPS


def _targets(fleet: Fleet, src: str, orders: tuple[Order, ...]) -> list[str]:
    """Бригады, в маршрут которых можно перенести заявки (без вывода новых бригад)."""
    ev = fleet.evaluator
    out = []
    for e in fleet.engineers:
        eid = e.id
        if eid == src:
            out.append(eid)
            continue
        if not fleet.available.get(eid, True) or not fleet.is_active(eid):
            continue
        if all(ev.pair_reason(e, o) is None for o in orders):
            out.append(eid)
    return out


def _relocate_pass(fleet: Fleet) -> int:
    """Перенос одной заявки в лучшую позицию любого маршрута (включая свой)."""
    ev = fleet.evaluator
    moves = 0
    for src_eng in fleet.engineers:
        src = src_eng.id
        i = 0
        while i < len(fleet.states[src]):
            st = fleet.states[src]
            order = st.stops[i].order
            rem = ev.without(st, i)
            if rem is None:
                i += 1
                continue
            crew_gain = 1 if len(rem) == 0 and not fleet.pinned.get(src, False) else 0
            gain = st.cost - rem.cost
            b_gain = st.breaches - rem.breaches
            best: tuple[Delta, str, int] | None = None
            for dst in _targets(fleet, src, (order,)):
                target = rem if dst == src else fleet.states[dst]
                ins = ev.best_insertion(target, order)
                if ins is None:
                    continue
                cand = (ins.breach_delta - b_gain, 0 if dst == src else -crew_gain, ins.cost - gain)
                if _improves(cand) and (best is None or cand < best[0]):
                    best = (cand, dst, ins.pos)
            if best is None:
                i += 1
                continue
            _, dst, pos = best
            if dst == src:
                fleet.states[src] = ev.insert(rem, order, pos)
            else:
                fleet.states[src] = rem
                fleet.states[dst] = ev.insert(fleet.states[dst], order, pos)
            moves += 1
    return moves


def _or_opt_pass(fleet: Fleet, lengths: tuple[int, ...] = (2, 3)) -> int:
    """Перенос цепочки из 2–3 подряд идущих заявок в другое место (свой или чужой маршрут)."""
    ev = fleet.evaluator
    moves = 0
    for src_eng in fleet.engineers:
        src = src_eng.id
        for length in lengths:
            i = 0
            while i + length <= len(fleet.states[src]):
                st = fleet.states[src]
                seg = st.orders[i : i + length]
                rem = ev.without(st, i, length)
                if rem is None:
                    i += 1
                    continue
                crew_gain = 1 if len(rem) == 0 and not fleet.pinned.get(src, False) else 0
                gain = st.cost - rem.cost
                b_gain = st.breaches - rem.breaches
                first, last = st.stops[i].node, st.stops[i + length - 1].node
                best: tuple[Delta, str, RouteState] | None = None
                for dst in _targets(fleet, src, seg):
                    target = rem if dst == src else fleet.states[dst]
                    t_orders = target.orders
                    crews_part = 0 if dst == src else -crew_gain
                    leg = ev.leg_fn(target.engineer)
                    seg_km = sum(
                        leg(st.stops[k].node, st.stops[k + 1].node)[0]
                        for k in range(i, i + length - 1)
                    )
                    for pos in range(len(t_orders) + 1):
                        if dst == src and pos == i:
                            continue
                        if crews_part == 0 and b_gain == 0:
                            # Вставка цепочки не уменьшает опоздания, нарушения SLA и смены зон
                            # в маршруте, поэтому прирост км — точная нижняя граница прироста
                            # стоимости, а ход улучшает план, только если стоимость падает.
                            if best is not None and best[0][:2] < (0, 0):
                                break
                            limit = -EPS if best is None else min(-EPS, best[0][2])
                            prev = target.start_node if pos == 0 else target.stops[pos - 1].node
                            lb = leg(prev, first)[0] + seg_km - gain
                            if pos < len(t_orders):
                                lb += leg(last, target.stops[pos].node)[0] - target.leg_km[pos]
                            if lb >= limit + 1e-9:
                                continue
                        new = ev.with_orders(target, t_orders[:pos] + seg + t_orders[pos:])
                        if new is None:
                            continue
                        cand = (
                            new.breaches - target.breaches - b_gain,
                            crews_part,
                            new.cost - target.cost - gain,
                        )
                        if _improves(cand) and (best is None or cand < best[0]):
                            best = (cand, dst, new)
                if best is None:
                    i += 1
                    continue
                _, dst, new = best
                if dst != src:
                    fleet.states[src] = rem
                fleet.states[dst] = new
                moves += 1
    return moves


def _replace_bound(leg: LegFn, st: RouteState, i: int, node: int, zone: str) -> float:
    """Нижняя граница прироста стоимости маршрута без аварий позже SLA, если i-ю заявку заменить
    заявкой в узле node: км и смены зон меняются только на соседних отрезках, а опоздания
    и нарушения SLA могут только появиться."""
    old = st.stops[i]
    prev_node = st.start_node if i == 0 else st.stops[i - 1].node
    dkm = leg(prev_node, node)[0] - st.leg_km[i]
    dsw = 0
    if i > 0:
        prev_zone = st.stops[i - 1].zone
        dsw += (prev_zone != zone) - (prev_zone != old.zone)
    if i + 1 < len(st):
        nxt = st.stops[i + 1]
        dkm += leg(node, nxt.node)[0] - st.leg_km[i + 1]
        dsw += (zone != nxt.zone) - (old.zone != nxt.zone)
    return dkm + obj.ZONE_SWITCH_KM * dsw


def _swap_pass(fleet: Fleet) -> int:
    """Обмен двумя заявками между маршрутами (каждая встаёт на место другой)."""
    ev = fleet.evaluator
    moves = 0
    eids = [e.id for e in fleet.engineers if fleet.is_active(e.id) and fleet.available.get(e.id, True)]
    for a_idx, a in enumerate(eids):
        for b in eids[a_idx + 1 :]:
            improved = True
            while improved:
                improved = False
                sa, sb = fleet.states[a], fleet.states[b]
                ao, bo = sa.orders, sb.orders
                # без нарушений SLA в обоих маршрутах ход улучшает план, только если дешевле
                bounded = sa.breaches == 0 and sb.breaches == 0
                leg_a, leg_b = ev.leg_fn(sa.engineer), ev.leg_fn(sb.engineer)
                for i, oi in enumerate(ao):
                    if ev.pair_reason(sb.engineer, oi) is not None:
                        continue
                    si = sa.stops[i]
                    for j, oj in enumerate(bo):
                        if ev.pair_reason(sa.engineer, oj) is not None:
                            continue
                        if bounded:
                            sj = sb.stops[j]
                            lb = _replace_bound(leg_a, sa, i, sj.node, sj.zone)
                            lb += _replace_bound(leg_b, sb, j, si.node, si.zone)
                            if lb >= -EPS + 1e-9:
                                continue
                        new_a = ev.with_orders(sa, ao[:i] + (oj,) + ao[i + 1 :])
                        if new_a is None:
                            continue
                        new_b = ev.with_orders(sb, bo[:j] + (oi,) + bo[j + 1 :])
                        if new_b is None:
                            continue
                        delta = (
                            new_a.breaches + new_b.breaches - sa.breaches - sb.breaches,
                            0,
                            new_a.cost + new_b.cost - sa.cost - sb.cost,
                        )
                        if _improves(delta):
                            fleet.states[a], fleet.states[b] = new_a, new_b
                            moves += 1
                            improved = True
                            break
                    if improved:
                        break
    return moves


def _two_opt_pass(fleet: Fleet) -> int:
    """Разворот участка внутри маршрута."""
    ev = fleet.evaluator
    moves = 0
    for eng in fleet.engineers:
        eid = eng.id
        improved = True
        while improved:
            improved = False
            st = fleet.states[eid]
            orders = st.orders
            n = len(orders)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new = ev.with_orders(st, orders[:i] + orders[i : j + 1][::-1] + orders[j + 1 :])
                    if new is not None and _improves(
                        (new.breaches - st.breaches, 0, new.cost - st.cost)
                    ):
                        fleet.states[eid] = new
                        moves += 1
                        improved = True
                        break
                if improved:
                    break
    return moves


def _two_opt_star_pass(fleet: Fleet) -> int:
    """Обмен хвостами двух маршрутов после выбранных позиций.

    В отличие от обычного 2-opt, маршруты не разворачиваются: хвосты просто меняются
    местами. Это позволяет убрать дорогой стык между двумя бригадами и одновременно
    учитывает навыки, транспорт, окна и смены через единый оценщик.
    """
    ev = fleet.evaluator
    moves = 0
    eids = [
        e.id
        for e in fleet.engineers
        if fleet.is_active(e.id) and fleet.available.get(e.id, True)
    ]
    for a_idx, a in enumerate(eids):
        for b in eids[a_idx + 1 :]:
            improved = True
            while improved:
                improved = False
                sa, sb = fleet.states[a], fleet.states[b]
                for i in range(1, len(sa)):
                    for j in range(1, len(sb)):
                        orders_a = sa.orders[:i] + sb.orders[j:]
                        orders_b = sb.orders[:j] + sa.orders[i:]
                        if not all(ev.pair_reason(sa.engineer, o) is None for o in orders_a):
                            continue
                        if not all(ev.pair_reason(sb.engineer, o) is None for o in orders_b):
                            continue
                        new_a = ev.with_orders(sa, orders_a)
                        if new_a is None:
                            continue
                        new_b = ev.with_orders(sb, orders_b)
                        if new_b is None:
                            continue
                        delta = (
                            new_a.breaches + new_b.breaches - sa.breaches - sb.breaches,
                            0,
                            new_a.cost + new_b.cost - sa.cost - sb.cost,
                        )
                        if _improves(delta):
                            fleet.states[a], fleet.states[b] = new_a, new_b
                            moves += 1
                            improved = True
                            break
                    if improved:
                        break
    return moves


def improve(fleet: Fleet, *, max_rounds: int = 60) -> int:
    """Применяет ходы, пока они улучшают план. Возвращает число принятых ходов."""
    total = 0
    for _ in range(max_rounds):
        moves = _relocate_pass(fleet)
        moves += _swap_pass(fleet)
        moves += _or_opt_pass(fleet)
        moves += _two_opt_star_pass(fleet)
        moves += _two_opt_pass(fleet)
        total += moves
        if moves == 0:
            break
    return total
