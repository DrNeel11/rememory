"""Real hardware detection for the first-run setup screen. No mocked values --
every field here is either a real subprocess/API call or explicitly None when
the corresponding hardware/service isn't present, so the UI can say "no NVIDIA
GPU detected" honestly instead of making something up.

GPU name/VRAM reuse scripts/ramer_local.py's nvidia-smi helpers rather than
reimplementing them -- same command, same parsing, one place to fix if it
ever breaks.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import urllib.request
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ramer_local import gpu_vram_mib, gpu_name  # noqa: E402  -- reuse, don't duplicate

OLLAMA_HOST = "http://127.0.0.1:11434"  # not "localhost" -- see ramer_local.py's OLLAMA_HOST comment


@dataclass
class GPUInfo:
    name: Optional[str] = None
    total_vram_mib: Optional[int] = None
    free_vram_mib: Optional[int] = None
    driver_version: Optional[str] = None


@dataclass
class InstalledModel:
    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    quantization: str = ""


@dataclass
class HardwareSnapshot:
    gpu: GPUInfo
    cuda_available: bool
    system_ram_gib: Optional[float]
    cpu_name: Optional[str]
    os_name: str
    ollama_available: bool
    installed_models: list[InstalledModel] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # what couldn't be detected and why -- surfaced, not swallowed


def detect_gpu() -> GPUInfo:
    try:
        total_mib, free_mib = gpu_vram_mib()
        name = gpu_name()
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return GPUInfo()  # no NVIDIA GPU, or nvidia-smi not on PATH -- both real, honest outcomes
    driver = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()
        driver = out[0].strip() if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return GPUInfo(name=name, total_vram_mib=total_mib, free_vram_mib=free_mib, driver_version=driver)


def detect_system_ram_gib() -> Optional[float]:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        return None


def detect_cpu_name() -> Optional[str]:
    """platform.processor() on Windows often returns a generic string like
    "AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD" instead of a real
    product name -- ask WMI for the actual name first, fall back to that
    generic string rather than failing outright."""
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                return out
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine() or None


def detect_os_name() -> str:
    return f"{platform.system()} {platform.release()}"


def _ollama_get(path: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}{path}", timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def detect_ollama() -> tuple[bool, list[InstalledModel]]:
    tags = _ollama_get("/api/tags")
    if tags is None:
        return False, []
    models = []
    for entry in tags.get("models", []):
        details = entry.get("details", {}) or {}
        models.append(InstalledModel(
            name=entry.get("name", ""),
            size_bytes=entry.get("size", 0),
            parameter_size=details.get("parameter_size", ""),
            quantization=details.get("quantization_level", ""),
        ))
    return True, models


def detect_hardware() -> HardwareSnapshot:
    errors = []
    gpu = detect_gpu()
    if gpu.name is None:
        errors.append("No NVIDIA GPU detected (nvidia-smi not found or returned no device).")
    ram = detect_system_ram_gib()
    if ram is None:
        errors.append("Could not read system RAM (psutil unavailable).")
    ollama_available, models = detect_ollama()
    if not ollama_available:
        errors.append("Ollama isn't running or isn't reachable at 127.0.0.1:11434.")
    return HardwareSnapshot(
        gpu=gpu,
        cuda_available=gpu.driver_version is not None,
        system_ram_gib=ram,
        cpu_name=detect_cpu_name(),
        os_name=detect_os_name(),
        ollama_available=ollama_available,
        installed_models=models,
        errors=errors,
    )
