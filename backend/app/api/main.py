import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.cli import REGIONS, load_normalized_dataset
from app.domain.enums import JobStatus, Skill, WorkType
from app.domain.events import EventType, PlanDiff, ReplanEvent
from app.domain.models import (
    Engineer,
    Location,
    Order,
    Plan,
    TimeWindow,
    minutes_to_time,
)
from app.solver import BaselineSolver, Solver, compare_plans
from app.solver.evaluator import RouteEvaluator
from app.solver.explain import ExplanationGenerator
from app.solver.replan import ReplanEngine, ReplanError
from app.solver.replan.alternatives import alternatives

app = FastAPI(
    title="MCT Dispatcher API",
    description="Интеллектуальный сервис планирования рабочих маршрутов инженеров (Билайн Бизнес x ЛЦТ 2026)",
    version="0.2.0",
)

# Состояние демо хранится в памяти одного процесса; сериализуем изменения, чтобы два
# одновременных события не прочитали одну версию плана и не затёрли результат друг друга.
state_lock = threading.RLock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class RegionState:
    """Сценарий дня по региону: утренний план, текущая версия после событий и журнал событий."""

    region: str
    orders: list[Order]
    engineers: list[Engineer]
    original: Plan
    baseline: Plan
    current: Plan
    original_orders: list[Order]
    events: list[PlanDiff] = field(default_factory=list)


@dataclass
class CustomDataset:
    """Набор, загруженный диспетчером (ТЗ §2.1.1): живёт в памяти сервера до перезапуска."""

    name: str
    orders: list[Order]
    engineers: list[Engineer]


STATE: dict[str, RegionState] = {}
CUSTOM: dict[str, CustomDataset] = {}
MAX_UPLOAD_ORDERS = 300
MAX_UPLOAD_ENGINEERS = 60
replan_engine = ReplanEngine()


class SolveRequest(BaseModel):
    region: str = Field(..., description="Идентификатор региона: east, southeast, southcenter")
    use_local_search: bool = Field(default=True, description="Использовать локальный поиск")


class SolveResponse(BaseModel):
    region: str
    region_name: str
    optimized: Plan
    baseline: Plan
    morning: Plan
    diff: dict[str, Any]
    orders: list[Order]
    engineers: list[Engineer]
    explanations: dict[str, str]
    baseline_explanations: dict[str, str] = Field(default_factory=dict)
    route_explanations: dict[str, str] = Field(default_factory=dict)
    replan_diff: PlanDiff | None = None
    events: list[PlanDiff] = Field(default_factory=list)


class DatasetResponse(BaseModel):
    region: str
    region_name: str
    orders: list[Order]
    engineers: list[Engineer]
    solved: bool


class CandidateOut(BaseModel):
    engineer_id: str
    engineer_name: str
    is_current: bool
    is_active: bool
    code: str | None
    reason: str | None
    delta_km: float | None = None
    start_time: str | None = None
    shift_min: int | None = None


class AlternativesResponse(BaseModel):
    order_id: str
    as_of_min: int | None
    locked: str | None
    current_engineer_id: str | None
    candidates: list[CandidateOut]


class UploadRequest(BaseModel):
    name: str = Field(default="Свой набор", max_length=60, description="Название набора в интерфейсе")
    orders: list[Order] = Field(..., description="Заявки в формате datasets/*.orders.json")
    engineers: list[Engineer] = Field(..., description="Бригады в формате datasets/*.engineers.json")


class ReplanEventRequest(BaseModel):
    region: str = Field(..., description="Идентификатор региона: east, southeast, southcenter")
    event: ReplanEvent = Field(..., description="Параметры события оперативного дня")


class ScenarioItem(BaseModel):
    id: str
    title: str
    event_type: EventType
    description: str
    event: ReplanEvent


def _region_key(region: str) -> str:
    reg = region.lower().strip()
    if reg not in REGIONS and reg not in CUSTOM:
        known = list(REGIONS) + list(CUSTOM)
        raise HTTPException(status_code=404, detail=f"Неизвестный регион: '{reg}'. Доступны: {known}")
    return reg


def _region_name(reg: str) -> str:
    return CUSTOM[reg].name if reg in CUSTOM else REGIONS[reg]["name_ru"]


def _load(reg: str) -> tuple[list[Order], list[Engineer]]:
    if reg in CUSTOM:
        ds = CUSTOM[reg]
        return list(ds.orders), list(ds.engineers)
    return load_normalized_dataset(reg)


def _response(st: RegionState) -> SolveResponse:
    return SolveResponse(
        region=st.region,
        region_name=_region_name(st.region),
        optimized=st.current,
        baseline=st.baseline,
        morning=st.original,
        diff=compare_plans(st.current, st.baseline),
        orders=st.orders,
        engineers=st.engineers,
        explanations=ExplanationGenerator.enrich_plan_explanations(
            st.current, st.orders, st.engineers, replan_engine.matrix
        ),
        baseline_explanations=ExplanationGenerator.enrich_plan_explanations(
            st.baseline, st.original_orders, st.engineers, replan_engine.matrix
        ),
        route_explanations=ExplanationGenerator.route_explanations(
            st.current, st.orders, st.engineers
        ),
        replan_diff=st.events[-1] if st.events else None,
        events=st.events,
    )


def _state(region: str) -> RegionState:
    reg = _region_key(region)
    if reg not in STATE:
        solve_plan(SolveRequest(region=reg))
    return STATE[reg]


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    """Возвращает список доступных регионов и базовую статистику."""
    datasets_meta = []
    for reg_key, reg_info in REGIONS.items():
        try:
            orders, engineers = load_normalized_dataset(reg_key)
        except (FileNotFoundError, ValueError, KeyError):
            orders, engineers = [], []
        datasets_meta.append(
            {
                "id": reg_key,
                "name": reg_info["name_ru"],
                "orders_count": len(orders),
                "engineers_count": len(engineers),
                "solved": reg_key in STATE,
            }
        )
    for reg_key, ds in CUSTOM.items():
        datasets_meta.append(
            {
                "id": reg_key,
                "name": ds.name,
                "orders_count": len(ds.orders),
                "engineers_count": len(ds.engineers),
                "solved": reg_key in STATE,
            }
        )
    return datasets_meta


@app.post("/api/datasets/upload")
def upload_dataset(request: UploadRequest) -> dict[str, Any]:
    """Загружает свой набор заявок и бригад (JSON в формате datasets/*.json) как новый регион."""
    orders, engineers = request.orders, request.engineers
    if not orders or not engineers:
        raise HTTPException(status_code=400, detail="Нужна хотя бы одна заявка и одна бригада")
    if len(orders) > MAX_UPLOAD_ORDERS or len(engineers) > MAX_UPLOAD_ENGINEERS:
        raise HTTPException(
            status_code=400,
            detail=f"Слишком большой набор: до {MAX_UPLOAD_ORDERS} заявок и {MAX_UPLOAD_ENGINEERS} бригад",
        )
    for kind, ids in (("заявок", [o.id for o in orders]), ("бригад", [e.id for e in engineers])):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        if dup:
            raise HTTPException(status_code=400, detail=f"Повторяются ID {kind}: {', '.join(dup[:5])}")
    for o in orders:
        if o.duration_min <= 0:
            raise HTTPException(
                status_code=400, detail=f"Заявка #{o.id}: окно или длительность заданы неверно"
            )
    for e in engineers:
        if e.shift.end_min <= e.shift.start_min:
            raise HTTPException(status_code=400, detail=f"Бригада {e.name}: смена задана неверно")
    reg = f"custom-{len(CUSTOM) + 1}"
    name = request.name.strip() or "Свой набор"
    with state_lock:
        CUSTOM[reg] = CustomDataset(name=name, orders=list(orders), engineers=list(engineers))
        STATE.pop(reg, None)
    return {
        "id": reg,
        "name": name,
        "orders_count": len(orders),
        "engineers_count": len(engineers),
        "solved": False,
    }


@app.get("/api/dataset/{region}", response_model=DatasetResponse)
def get_dataset(region: str) -> DatasetResponse:
    """Исходные данные региона без расчёта: заявки и бригады (шаг 1 демо — «открыть набор»)."""
    reg = _region_key(region)
    orders, engineers = _load(reg)
    return DatasetResponse(
        region=reg,
        region_name=_region_name(reg),
        orders=orders,
        engineers=engineers,
        solved=reg in STATE,
    )


@app.post("/api/plan/solve", response_model=SolveResponse)
def solve_plan(request: SolveRequest) -> SolveResponse:
    """Строит утренний план региона и базовый вариант FIFO; сбрасывает события дня."""
    with state_lock:
        reg = _region_key(request.region)
        orders, engineers = _load(reg)
        solver = Solver(replan_engine.matrix, use_local_search=request.use_local_search)
        plan = solver.solve(orders, engineers)
        baseline = BaselineSolver(replan_engine.matrix).solve(orders, engineers)
        STATE[reg] = RegionState(
            region=reg,
            orders=list(orders),
            engineers=engineers,
            original=plan,
            baseline=baseline,
            current=plan,
            original_orders=list(orders),
        )
        return _response(STATE[reg])


@app.get("/api/plan/{region}", response_model=SolveResponse)
def get_plan(region: str) -> SolveResponse:
    """Текущая версия плана региона (с учётом применённых событий)."""
    with state_lock:
        return _response(_state(region))


@app.post("/api/plan/event")
def apply_replan_event(request: ReplanEventRequest) -> dict[str, Any]:
    """Применяет событие дня (новая заявка, авария, отмена, сход бригады) к текущему плану."""
    with state_lock:
        st = _state(request.region)
        try:
            new_plan, plan_diff, orders = replan_engine.apply_event(
                st.current, request.event, st.orders, st.engineers
            )
        except ReplanError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        st.current = new_plan
        st.orders = orders
        st.events.append(plan_diff)
        resp = _response(st)
        return {
            "region": st.region,
            "optimized": resp.optimized,
            "morning": resp.morning,
            "diff": resp.diff,
            "plan_diff": plan_diff,
            "orders": resp.orders,
            "engineers": resp.engineers,
            "explanations": resp.explanations,
            "route_explanations": resp.route_explanations,
            "events": resp.events,
        }


@app.post("/api/plan/reset/{region}", response_model=SolveResponse)
def reset_plan(region: str) -> SolveResponse:
    """Возвращает утренний план: отменяет все события дня."""
    with state_lock:
        reg = _region_key(region)
        if reg not in STATE:
            return solve_plan(SolveRequest(region=reg))
        st = STATE[reg]
        st.current = st.original
        st.orders = list(st.original_orders)
        st.events = []
        return _response(st)


def _next_slot(t: int) -> int:
    """Начало ближайшего двухчасового окна с чётного часа не раньше t + 1 ч."""
    start = ((t + 60 + 119) // 120) * 120
    return min(start, 20 * 60)


@app.get("/api/scenarios/{region}")
def get_scenarios(region: str) -> list[ScenarioItem]:
    """Готовые сценарии событий для демонстрации (ТЗ §4) по текущему плану региона."""
    st = _state(region)
    plan, orders, engineers = st.current, st.orders, st.engineers
    orders_map = {o.id: o for o in orders}
    eng_map = {e.id: e for e in engineers}
    as_of = plan.as_of_min or 0
    scenarios: list[ScenarioItem] = []

    def at(default: int) -> int:
        return max(default, as_of)

    # точка события — рядом с реальной заявкой региона (медианная по широте)
    ref = sorted(orders, key=lambda o: (o.location.lat, o.id))[len(orders) // 2].location
    near = Location(
        lat=round(ref.lat + 0.003, 5),
        lon=round(ref.lon - 0.004, 5),
        address=f"{ref.address} (соседний дом)",
        district=ref.district,
    )

    t = at(11 * 60)
    emergency = Order(
        id=f"AV-{minutes_to_time(t).replace(':', '')}",
        skills=[Skill.EMERGENCY],
        work_type=WorkType.EMERGENCY,
        window=TimeWindow(start=minutes_to_time(t), end="23:59"),
        duration_min=WorkType.EMERGENCY.work_min,
        location=near,
    )
    scenarios.append(
        ScenarioItem(
            id="scenario-emergency",
            title=f"Авария в {minutes_to_time(t)}",
            event_type=EventType.URGENT_ORDER,
            description=(
                f"Нет связи у клиентов в районе {near.district}. Норматив 80 мин работ, начать как можно "
                "раньше (ориентир — 2 ч). Начатые работы не прерываются, хвост дня перестраивается."
            ),
            event=ReplanEvent(
                event_type=EventType.URGENT_ORDER,
                event_time=minutes_to_time(t),
                new_order=emergency,
                description="Авария: нет связи у абонентов",
            ),
        )
    )

    t = at(12 * 60)
    slot = _next_slot(t)
    connection = Order(
        id=f"NEW-{minutes_to_time(t).replace(':', '')}",
        skills=[Skill.CONNECTION],
        work_type=WorkType.CONNECTION,
        window=TimeWindow(start=minutes_to_time(slot), end=minutes_to_time(slot + 120)),
        duration_min=WorkType.CONNECTION.work_min,
        location=Location(
            lat=round(ref.lat - 0.004, 5),
            lon=round(ref.lon + 0.005, 5),
            address=f"{ref.address} (новый абонент)",
            district=ref.district,
        ),
    )
    scenarios.append(
        ScenarioItem(
            id="scenario-new-order",
            title=f"Новое подключение в {minutes_to_time(t)}",
            event_type=EventType.NEW_ORDER,
            description=(
                f"Клиент записался на окно {connection.window.start}–{connection.window.end}. "
                "Заявка встаёт в свободный интервал, план остальных бригад не перестраивается."
            ),
            event=ReplanEvent(
                event_type=EventType.NEW_ORDER,
                event_time=minutes_to_time(t),
                new_order=connection,
                description="Новая заявка на подключение",
            ),
        )
    )

    # отмена, когда бригада уже в пути (как в контрольных данных), иначе — запланированная
    t = at(13 * 60)
    target = None
    for route in plan.routes:
        for job in route.jobs:
            if job.status == JobStatus.CANCELLED or job.is_frozen:
                continue
            if job.departure_time_min is not None and job.departure_time_min <= t < job.arrival_time_min:
                target = (job, route.engineer_id, "в пути")
                break
        if target:
            break
    if target is None:
        for route in plan.routes:
            job = next((j for j in route.jobs if not j.is_frozen and j.start_time_min >= t + 60), None)
            if job is not None:
                target = (job, route.engineer_id, "запланирована")
                break
    if target is not None:
        job, eid, state = target
        order = orders_map[job.order_id]
        scenarios.append(
            ScenarioItem(
                id="scenario-cancel",
                title=f"Отмена заявки #{order.id} в {minutes_to_time(t)}",
                event_type=EventType.CANCEL_ORDER,
                description=(
                    f"{order.kind.label_ru}, {eng_map[eid].name} ({state}). Клиент отказался — "
                    "бригада освобождается, в свободное время ставим неназначенные заявки."
                ),
                event=ReplanEvent(
                    event_type=EventType.CANCEL_ORDER,
                    event_time=minutes_to_time(t),
                    order_id=order.id,
                    description="Клиент отказался от визита",
                ),
            )
        )

    t = at(14 * 60)
    busiest = max(
        (r for r in plan.routes if r.unavailable_from_min is None),
        key=lambda r: (sum(1 for j in r.jobs if j.start_time_min > t and not j.is_frozen), r.engineer_id),
        default=None,
    )
    if busiest is not None:
        eng = eng_map[busiest.engineer_id]
        left = sum(1 for j in busiest.jobs if j.start_time_min > t and not j.is_frozen)
        scenarios.append(
            ScenarioItem(
                id="scenario-breakdown",
                title=f"Сход бригады в {minutes_to_time(t)}: {eng.name}",
                event_type=EventType.ENGINEER_UNAVAILABLE,
                description=(
                    f"Поломка транспорта. Выполненные и начатые работы остаются за бригадой, "
                    f"оставшиеся {left} заявок распределяются по другим бригадам."
                ),
                event=ReplanEvent(
                    event_type=EventType.ENGINEER_UNAVAILABLE,
                    event_time=minutes_to_time(t),
                    engineer_id=eng.id,
                    description="Поломка транспорта",
                ),
            )
        )
    return scenarios


@app.get("/api/alternatives/{region}/{order_id}", response_model=AlternativesResponse)
def get_alternatives(region: str, order_id: str) -> AlternativesResponse:
    """Кто ещё мог бы взять заявку в текущем плане: прирост км и время начала или причина отказа."""
    st = _state(region)
    order = next((o for o in st.orders if o.id == order_id), None)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Заявка #{order_id} не найдена")
    ev = RouteEvaluator(
        replan_engine.matrix,
        day_start_min=min((e.shift.start_min for e in st.engineers), default=600),
    )
    alt = alternatives(ev, st.current, order, st.orders, st.engineers)
    return AlternativesResponse(
        order_id=alt.order_id,
        as_of_min=alt.as_of_min,
        locked=alt.locked,
        current_engineer_id=alt.current_engineer_id,
        candidates=[
            CandidateOut(
                engineer_id=c.engineer_id,
                engineer_name=c.engineer_name,
                is_current=c.is_current,
                is_active=c.is_active,
                code=c.code.value if c.code else None,
                reason=c.reason,
                delta_km=c.delta_km,
                start_time=c.start_time,
                shift_min=c.shift_min,
            )
            for c in alt.candidates
        ],
    )


@app.get("/api/explain/{order_id}")
def get_explanation(
    order_id: str,
    region: str = Query(default="east", description="Регион заявки"),
) -> dict[str, str]:
    """Возвращает детальное человекочитаемое объяснение назначения или неназначения заявки."""
    explanations = _response(_state(region)).explanations
    if order_id in explanations:
        return {"order_id": order_id, "explanation": explanations[order_id]}
    raise HTTPException(status_code=404, detail=f"Заявка #{order_id} не найдена в регионе {region}")


# Монтирование собранного React frontend (для демонстрации в один клик)
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
_dist_dir = _repo_root / "frontend" / "dist"

if _dist_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="frontend")
