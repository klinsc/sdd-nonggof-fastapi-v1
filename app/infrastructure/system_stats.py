"""System resource & GPU stats — used by /stats and the /qa/stream `stats` event.

NVML is initialized lazily on first use and reused across calls. If the
host has no NVIDIA driver / NVML lib, GPU calls return [] and the service
keeps working.
"""
from __future__ import annotations

import logging

import psutil

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - import-time only
    _NVML_AVAILABLE = False

logger = logging.getLogger(__name__)
_NVML_INITIALIZED = False


def _ensure_nvml() -> bool:
    global _NVML_INITIALIZED
    if not _NVML_AVAILABLE:
        return False
    if not _NVML_INITIALIZED:
        try:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        except Exception as exc:
            logger.warning("NVML init failed; GPU stats disabled (%s)", exc)
            return False
    return True


def get_cpu_stats() -> dict:
    return {
        "percent": float(psutil.cpu_percent(interval=None)),
        "count": int(psutil.cpu_count(logical=True) or 0),
    }


def get_memory_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "percent": round(float(vm.percent), 1),
    }


def get_gpu_stats() -> list[dict]:
    if not _ensure_nvml():
        return []
    out: list[dict] = []
    try:
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = None
            out.append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(h),
                "utilization_percent": float(util.gpu),
                "memory_used_gb": round(mem.used / (1024 ** 3), 2),
                "memory_total_gb": round(mem.total / (1024 ** 3), 2),
                "memory_percent": round(mem.used / mem.total * 100, 1),
                "temperature_c": temp,
            })
    except Exception as exc:
        logger.warning("NVML read failed: %s", exc)
    return out


def shutdown_nvml() -> None:
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _NVML_INITIALIZED = False
