"""GET /stats — system resource snapshot for monitoring dashboards."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.dependencies import require_api_key
from app.infrastructure import system_stats

router = APIRouter(
    prefix="/stats",
    tags=["stats"],
    dependencies=[Depends(require_api_key)],
)


class CpuStats(BaseModel):
    percent: float = Field(..., description="System-wide CPU utilization (0-100).")
    count: int = Field(..., description="Logical CPU count.")


class MemoryStats(BaseModel):
    used_gb: float
    total_gb: float
    percent: float


class GpuStats(BaseModel):
    index: int
    name: str
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    temperature_c: float | None = Field(default=None, description="None if the driver does not expose temperature.")


class StatsResponse(BaseModel):
    cpu: CpuStats
    memory: MemoryStats
    gpus: list[GpuStats] = Field(
        default_factory=list,
        description="Empty list if NVML is unavailable (no NVIDIA driver, or non-GPU host).",
    )
    llm_profile: str
    llm_model: str
    embedding_profile: str
    embedding_model: str
    retrieval_k: int
    retrieval_score_threshold: float


@router.get(
    "",
    response_model=StatsResponse,
    summary="Current system + configuration snapshot",
    description=(
        "Returns a point-in-time snapshot of CPU, RAM, and per-GPU utilization, "
        "plus the active LLM/embedding profile and retrieval settings. "
        "Cheap to call (~5–10 ms) — safe to poll for dashboards."
    ),
)
async def get_stats(settings: Settings = Depends(get_settings)) -> StatsResponse:
    return StatsResponse(
        cpu=CpuStats(**system_stats.get_cpu_stats()),
        memory=MemoryStats(**system_stats.get_memory_stats()),
        gpus=[GpuStats(**g) for g in system_stats.get_gpu_stats()],
        llm_profile=settings.LLM_PROFILE,
        llm_model=settings.LLM_MODEL,
        embedding_profile=settings.EMBEDDING_PROFILE,
        embedding_model=settings.EMBEDDING_MODEL,
        retrieval_k=settings.RETRIEVAL_K,
        retrieval_score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
    )
