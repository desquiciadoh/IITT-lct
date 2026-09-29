import math
import re
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from app.domain.enums import (
    JobStatus,
    OrderStatus,
    Priority,
    ReasonCode,
    Skill,
    Transport,
    WorkType,
)


def time_to_minutes(t_str: str) -> int:
    """Конвертация времени HH:MM в минуты от начала суток."""
    value = t_str.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", value) is None:
        raise ValueError(f"Invalid time format: {t_str}, expected HH:MM")
    hours, minutes = (int(part) for part in value.split(":"))
    if hours > 23 or minutes > 59:
        raise ValueError(f"Invalid time value: {t_str}, expected HH:MM within one day")
    return hours * 60 + minutes


def minutes_to_time(minutes: int) -> str:
    """Конвертация минут от начала суток в формат HH:MM."""
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


class TimeWindow(BaseModel):
    """Временное окно визита или смены инженера."""

    start: str = Field(..., description="Время начала в формате HH:MM")
    end: str = Field(..., description="Время окончания в формате HH:MM")
    start_min: int = Field(default=0)
    end_min: int = Field(default=0)

    @model_validator(mode="before")
    @classmethod
    def calculate_minutes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            s = data.get("start")
            e = data.get("end")
            if s:
                data["start_min"] = time_to_minutes(s)
            if e:
                data["end_min"] = time_to_minutes(e)
        return data

    @model_validator(mode="after")
    def validate_window(self) -> "TimeWindow":
        if self.end_min < self.start_min:
            raise ValueError("Time window end must not be earlier than start")
        return self

    def contains(self, minute: int) -> bool:
        """Попадает ли минута внутрь окна [start, end]."""
        return self.start_min <= minute <= self.end_min

    def overlaps(self, other: "TimeWindow") -> bool:
        """Пересекаются ли два интервала."""
        return max(self.start_min, other.start_min) <= min(self.end_min, other.end_min)


class Location(BaseModel):
    """Географическая точка заявки или депо."""

    lat: float
    lon: float
    address: str
    district: str

    @model_validator(mode="after")
    def validate_coordinates(self) -> "Location":
        if not math.isfinite(self.lat) or not -90 <= self.lat <= 90:
            raise ValueError("Latitude must be finite and between -90 and 90")
        if not math.isfinite(self.lon) or not -180 <= self.lon <= 180:
            raise ValueError("Longitude must be finite and between -180 and 180")
        return self


class TechInfo(BaseModel):
    """Технические параметры подключения (доп. усложнение)."""

    product: str | None = None  # "FMC" или "FTTB"
    gbit: bool = False


class Order(BaseModel):
    """Заявка на проведение работ."""

    id: str = Field(..., min_length=1)
    skills: list[Skill]
    priority: Priority = Priority.NORMAL
    work_type: WorkType | None = Field(
        default=None, description="Вид работ BK; если не задан — выводится из навыка и длительности"
    )
    window: TimeWindow
    duration_min: int = Field(..., description="Длительность проведения работ в минутах")
    required_transport: Transport | None = None
    location: Location
    tech: TechInfo | None = None
    bk_type: str | None = None
    hd_type: str | None = None
    released_min: int | None = Field(
        default=None,
        description="Когда заявка поступила (минуты от полуночи). None — известна с начала дня",
    )
    covers: list[str] = Field(
        default_factory=list,
        description="Авария на узле: ID других заявок из файла, которые закрывает этот выезд",
    )
    status: OrderStatus = OrderStatus.UNASSIGNED

    @model_validator(mode="after")
    def infer_work_type(self) -> "Order":
        if self.work_type is None:
            if self.priority == Priority.URGENT or Skill.EMERGENCY in self.skills:
                self.work_type = WorkType.EMERGENCY
            elif Skill.CONNECTION in self.skills:
                self.work_type = WorkType.ADDON if self.duration_min <= 20 else WorkType.CONNECTION
            else:
                self.work_type = WorkType.LOCAL
        return self

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Order):
            return self.id == other.id
        return False

    @property
    def primary_skill(self) -> Skill:
        return self.skills[0] if self.skills else Skill.LOCAL

    @property
    def kind(self) -> WorkType:
        return self.work_type or WorkType.LOCAL

    @property
    def is_emergency(self) -> bool:
        return self.kind == WorkType.EMERGENCY


class Engineer(BaseModel):
    """Инженер / выездная бригада."""

    id: str = Field(..., min_length=1)
    name: str
    skills: list[Skill] = Field(..., min_length=1, max_length=3)
    transport: Transport
    shift: TimeWindow
    depot: Location

    def has_skill(self, skill: Skill) -> bool:
        return skill in self.skills


class AssignedJob(BaseModel):
    """Визит бригады к клиенту в маршруте."""

    order_id: str
    departure_time_min: int | None = Field(
        default=None, description="Выезд к клиенту; бригада выезжает так, чтобы прибыть к началу окна"
    )
    arrival_time_min: int
    start_time_min: int
    end_time_min: int
    travel_time_min: int
    travel_dist_km: float
    waiting_time_min: int = Field(default=0, description="Простой в предыдущей точке перед выездом")
    status: JobStatus = JobStatus.PLANNED
    is_frozen: bool = Field(
        default=False, description="Заморожена: бригада уже выехала, работает или закончила"
    )

    @model_validator(mode="after")
    def fill_departure(self) -> "AssignedJob":
        if self.departure_time_min is None:
            self.departure_time_min = self.arrival_time_min - self.travel_time_min
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def departure_time(self) -> str:
        return minutes_to_time(self.departure_time_min or 0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def arrival_time(self) -> str:
        return minutes_to_time(self.arrival_time_min)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def start_time(self) -> str:
        return minutes_to_time(self.start_time_min)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def end_time(self) -> str:
        return minutes_to_time(self.end_time_min)

    @property
    def counts_as_assigned(self) -> bool:
        return self.status != JobStatus.CANCELLED


class Route(BaseModel):
    """Маршрут одного инженера на рабочий день."""

    engineer_id: str
    jobs: list[AssignedJob] = Field(default_factory=list)
    total_distance_km: float = 0.0
    total_travel_time_min: int = 0
    total_work_time_min: int = 0
    total_waiting_time_min: int = 0
    unavailable_from_min: int | None = Field(
        default=None, description="С какого момента бригада сошла с линии (None — работает)"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_active(self) -> bool:
        return len(self.jobs) > 0


class UnassignedInfo(BaseModel):
    """Почему заявка не назначена: одна фраза для диспетчера и разбивка по бригадам."""

    code: ReasonCode
    text: str
    hint: str | None = None
    engineer_codes: dict[str, int] = Field(
        default_factory=dict, description="Код причины -> число бригад с этой причиной"
    )
    context: str | None = Field(default=None, description="Событие, из-за которого заявка снята")


class PlanMetrics(BaseModel):
    """Сводные метрики эффективности плана (ТЗ §2.3)."""

    total_orders: int
    assigned_orders: int
    unassigned_orders: int
    cancelled_orders: int = 0
    assignment_rate_pct: float  # доля назначенных среди не отменённых заявок

    active_engineers_count: int  # Количество задействованных бригад (>= 1 заявки)
    total_engineers_count: int

    total_distance_km: float  # Суммарный пробег
    car_distance_km: float = 0.0  # Пробег на автомобилях
    total_travel_time_min: int  # Суммарное время в пути
    total_work_time_min: int

    emergency_orders: int = 0
    emergency_within_sla: int = 0
    emergency_avg_reaction_min: float | None = None
    emergency_max_reaction_min: int | None = None

    extra_crews_needed: int = Field(
        default=0, description="Сколько ещё бригад нужно, чтобы закрыть неназначенные заявки"
    )

    avg_load_pct: float | None = Field(
        default=None, description="Загрузка бригад на линии: (дорога + работа) / длина смен, %"
    )
    low_load_crews: int = Field(
        default=0, description="Бригад на линии с загрузкой ниже 50 % (ориентир организаторов)"
    )

    engineer_distances: dict[str, float] = Field(default_factory=dict)  # ID бригады -> км
    engineer_order_counts: dict[str, int] = Field(default_factory=dict)  # ID бригады -> заявок


class Plan(BaseModel):
    """Итоговый план распределения работ на день."""

    routes: list[Route]
    unassigned_orders: dict[str, str] = Field(
        default_factory=dict, description="ID неназначенной заявки -> человекочитаемая причина"
    )
    unassigned_details: dict[str, UnassignedInfo] = Field(default_factory=dict)
    cancelled_orders: dict[str, str] = Field(
        default_factory=dict, description="ID отменённой заявки -> когда и как отменена"
    )
    metrics: PlanMetrics
    as_of_min: int | None = Field(
        default=None, description="Время последнего учтённого события; раньше него план не меняется"
    )
