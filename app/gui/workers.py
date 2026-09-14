"""QThread workers so hardware detection and model pulls never block the UI
thread. Hardware detection itself is fast (a few subprocess calls), but a
model pull can run for minutes, and Qt's event loop must keep processing
paint/input events the whole time or the window looks frozen -- the exact
"is it hung or just working" problem this project already hit once with
mini_agent.py's streaming, solved here by construction instead of by
timeout tuning.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from pathlib import Path
from typing import Callable, Optional

from app.core.agent import AgentEvent, AgentSession
from app.core.hardware import HardwareSnapshot, detect_hardware
from app.core.model_runtime import OllamaRuntime
from app.core.placement import PlacementEngine, PlacementResult


class HardwareDetectWorker(QThread):
    finished_ok = Signal(object)  # HardwareSnapshot

    def run(self) -> None:
        snapshot: HardwareSnapshot = detect_hardware()
        self.finished_ok.emit(snapshot)


class ModelPullWorker(QThread):
    progress = Signal(dict)  # raw Ollama pull chunk
    finished_ok = Signal(bool)

    def __init__(self, model: str, parent=None):
        super().__init__(parent)
        self.model = model

    def run(self) -> None:
        runtime = OllamaRuntime()
        ok = runtime.pull(self.model, on_progress=lambda chunk: self.progress.emit(chunk))
        self.finished_ok.emit(ok)


class PlacementWorker(QThread):
    log_line = Signal(str)
    finished_ok = Signal(object)  # PlacementResult

    def __init__(self, model: str, num_ctx: int, force: bool = False, parent=None):
        super().__init__(parent)
        self.model = model
        self.num_ctx = num_ctx
        self.force = force

    def run(self) -> None:
        engine = PlacementEngine()
        result: PlacementResult = engine.probe_model(
            self.model, num_ctx=self.num_ctx, force=self.force, on_log=lambda msg: self.log_line.emit(str(msg)),
        )
        self.finished_ok.emit(result)


class AgentWorker(QThread):
    event = Signal(object)  # AgentEvent
    finished_ok = Signal(object)  # dict result

    def __init__(self, model: str, workspace: Path, num_gpu, num_ctx: int, task: str,
                 permission_callback: Optional[Callable[[str, dict], bool]] = None,
                 baseline_tok_s: Optional[float] = None, parent=None):
        super().__init__(parent)
        self.session = AgentSession(
            model=model, workspace=workspace, num_gpu=num_gpu, num_ctx=num_ctx,
            permission_callback=permission_callback, baseline_tok_s=baseline_tok_s,
        )
        self.task = task

    def cancel(self) -> None:
        self.session.cancel()

    def run(self) -> None:
        result = self.session.run(self.task, on_event=lambda e: self.event.emit(e))
        self.finished_ok.emit(result)
