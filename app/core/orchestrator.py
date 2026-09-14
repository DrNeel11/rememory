"""The Orchestrator is the one object the GUI and the CLI both depend on.
Neither should call OllamaRuntime, PlacementEngine, or AgentSession
directly -- that's the "CLI must not contain separate business logic"
requirement from the spec, enforced by having exactly one place that
wires the pieces together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from app.core.agent import AgentSession
from app.core.hardware import HardwareSnapshot, detect_hardware
from app.core.model_runtime import CURATED_MODELS, ModelInfo, OllamaRuntime, recommend_model
from app.core.observability import log_session
from app.core.placement import PlacementEngine, PlacementResult
from app.core.proxy_server import DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM, RamerProxyServer


class Orchestrator:
    def __init__(self):
        self.model_runtime = OllamaRuntime()
        self.placement_engine = PlacementEngine()
        self.hardware: Optional[HardwareSnapshot] = None

    def detect_hardware(self) -> HardwareSnapshot:
        self.hardware = detect_hardware()
        return self.hardware

    def recommended_model(self) -> str:
        vram = self.hardware.gpu.total_vram_mib if self.hardware else None
        return recommend_model(vram)

    def curated_models(self) -> list[dict]:
        return CURATED_MODELS

    def list_installed_models(self) -> list[ModelInfo]:
        return self.model_runtime.list_installed()

    def pull_model(self, model: str, on_progress: Optional[Callable[[dict], None]] = None) -> bool:
        return self.model_runtime.pull(model, on_progress=on_progress)

    def optimize(self, model: str, num_ctx: int, force: bool = False,
                 on_log: Optional[Callable[[str], None]] = None) -> PlacementResult:
        return self.placement_engine.probe_model(model, num_ctx=num_ctx, force=force, on_log=on_log)

    def start_agent(self, model: str, workspace: Path, num_gpu, num_ctx: int,
                     permission_callback: Optional[Callable[[str, dict], bool]] = None,
                     baseline_tok_s: Optional[float] = None) -> AgentSession:
        return AgentSession(
            model=model, workspace=workspace, num_gpu=num_gpu, num_ctx=num_ctx,
            permission_callback=permission_callback, baseline_tok_s=baseline_tok_s,
        )

    def start_proxy_server(self, port: int = DEFAULT_PROXY_PORT, upstream: str = DEFAULT_UPSTREAM,
                            on_log: Optional[Callable[[str], None]] = None) -> RamerProxyServer:
        return RamerProxyServer(upstream=upstream, port=port, on_log=on_log)

    def run_agent_task(self, session: AgentSession, task: str, on_event: Callable) -> dict:
        result = session.run(task, on_event)
        log_session(
            model=session.model, num_ctx=session.num_ctx, num_gpu=session.num_gpu,
            turns=session.turns, result=result, hardware=self.hardware,
        )
        return result
