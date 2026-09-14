"""ModelRuntime abstraction: the rest of the application talks to this
interface, never to Ollama's HTTP API directly. OllamaRuntime is the only
implementation today; a future LlamaCppRuntime or MLXRuntime slots in here
without the orchestrator, GUI, or CLI knowing the difference.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ramer_local import OLLAMA_HOST  # noqa: E402 -- reuse the 127.0.0.1 host constant, not "localhost"


@dataclass
class ModelInfo:
    name: str
    installed: bool
    size_bytes: int = 0
    parameter_size: str = ""
    quantization: str = ""
    block_count: Optional[int] = None


# Small, deliberately curated set for V1 -- every one of these has already
# been measured on real hardware in this project's own research
# (PRODUCT_HYPOTHESES.md), so "recommended" here means "known to work",
# not a guess. Expanding this list is a later, explicit decision.
CURATED_MODELS = [
    {"name": "qwen2.5:3b-instruct", "stars": 3, "note": "Fastest, runs on almost anything. Weakest coding ability."},
    {"name": "qwen2.5:7b-instruct", "stars": 4, "note": "Good balance for 6-8GB GPUs."},
    {"name": "qwen2.5:14b-instruct", "stars": 5, "note": "Best coding quality this app has tuned for. Needs ~7GB+ VRAM to place well."},
]


class ModelRuntime(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def list_installed(self) -> list[ModelInfo]: ...

    @abstractmethod
    def is_installed(self, model: str) -> bool: ...

    @abstractmethod
    def pull(self, model: str, on_progress: Optional[Callable[[dict], None]] = None) -> bool: ...

    @abstractmethod
    def get_metadata(self, model: str) -> Optional[ModelInfo]: ...

    @abstractmethod
    def unload(self, model: str) -> None: ...


class OllamaRuntime(ModelRuntime):
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host

    def _get(self, path: str, timeout: float = 5.0) -> Optional[dict]:
        try:
            with urllib.request.urlopen(f"{self.host}{path}", timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def available(self) -> bool:
        return self._get("/api/tags", timeout=3.0) is not None

    def list_installed(self) -> list[ModelInfo]:
        tags = self._get("/api/tags")
        if tags is None:
            return []
        out = []
        for entry in tags.get("models", []):
            details = entry.get("details", {}) or {}
            out.append(ModelInfo(
                name=entry.get("name", ""), installed=True, size_bytes=entry.get("size", 0),
                parameter_size=details.get("parameter_size", ""), quantization=details.get("quantization_level", ""),
            ))
        return out

    def is_installed(self, model: str) -> bool:
        return any(m.name == model for m in self.list_installed())

    def pull(self, model: str, on_progress: Optional[Callable[[dict], None]] = None) -> bool:
        """Streams Ollama's own pull progress (status, completed/total bytes)
        back through on_progress so the UI shows real download progress, not
        a spinner. Returns True only if Ollama's final chunk reports success."""
        req = urllib.request.Request(
            f"{self.host}/api/pull",
            data=json.dumps({"model": model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                success = False
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if on_progress:
                        on_progress(chunk)
                    if chunk.get("status") == "success":
                        success = True
                return success
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return False

    def get_metadata(self, model: str) -> Optional[ModelInfo]:
        show = self._get_post("/api/show", {"model": model})
        if show is None:
            return None
        details = show.get("details", {}) or {}
        model_info = show.get("model_info", {}) or {}
        block_count = None
        for key, value in model_info.items():
            if key.endswith(".block_count"):
                block_count = int(value)
                break
        return ModelInfo(
            name=model, installed=self.is_installed(model),
            parameter_size=details.get("parameter_size", ""), quantization=details.get("quantization_level", ""),
            block_count=block_count,
        )

    def _get_post(self, path: str, payload: dict, timeout: float = 10.0) -> Optional[dict]:
        req = urllib.request.Request(
            f"{self.host}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def unload(self, model: str) -> None:
        self._get_post("/api/generate", {"model": model, "keep_alive": 0})


def recommend_model(vram_mib: Optional[int]) -> str:
    """Simple, explainable recommendation, not a learned model: pick the
    largest curated model this project has actually validated as fitting
    the detected VRAM. Deliberately conservative -- a wrong-but-safe
    recommendation beats a confident guess with no data behind it."""
    if vram_mib is None or vram_mib < 5000:
        return "qwen2.5:3b-instruct"
    if vram_mib < 9000:
        return "qwen2.5:7b-instruct"
    return "qwen2.5:14b-instruct"
