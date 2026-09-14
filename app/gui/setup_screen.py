from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame

from app.core.hardware import HardwareSnapshot
from app.gui.workers import HardwareDetectWorker


class SetupScreen(QWidget):
    continue_clicked = Signal(HardwareSnapshot)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot: HardwareSnapshot | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Welcome")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel("Let's set up your local AI.")
        subtitle.setStyleSheet("color: #888;")
        layout.addWidget(subtitle)

        self.status_label = QLabel("Detecting hardware...")
        self.status_label.setStyleSheet("margin-top: 16px; font-style: italic; color: #888;")
        layout.addWidget(self.status_label)

        self.result_frame = QFrame()
        self.result_layout = QVBoxLayout(self.result_frame)
        layout.addWidget(self.result_frame)

        layout.addStretch()

        self.continue_button = QPushButton("Continue")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self._on_continue)
        layout.addWidget(self.continue_button)

        self._worker = HardwareDetectWorker()
        self._worker.finished_ok.connect(self._on_detected)
        self._worker.start()

    def _row(self, label: str, value: str, ok: bool | None = None) -> QLabel:
        prefix = "" if ok is None else ("✓ " if ok else "✗ ")
        color = "" if ok is None else ("color: #4a9; " if ok else "color: #c44; ")
        w = QLabel(f"{prefix}{label}: {value}")
        w.setStyleSheet(f"{color}font-size: 13px;")
        return w

    def _on_detected(self, snapshot: HardwareSnapshot) -> None:
        self.snapshot = snapshot
        self.status_label.setText("")

        gpu = snapshot.gpu
        if gpu.name:
            self.result_layout.addWidget(self._row("GPU", f"{gpu.name} ({gpu.total_vram_mib} MiB VRAM)", ok=True))
        else:
            self.result_layout.addWidget(self._row("GPU", "No NVIDIA GPU detected", ok=False))

        self.result_layout.addWidget(self._row(
            "System RAM", f"{snapshot.system_ram_gib} GB" if snapshot.system_ram_gib else "unknown",
            ok=snapshot.system_ram_gib is not None,
        ))
        self.result_layout.addWidget(self._row("CPU", snapshot.cpu_name or "unknown", ok=snapshot.cpu_name is not None))
        self.result_layout.addWidget(self._row("OS", snapshot.os_name, ok=True))
        self.result_layout.addWidget(self._row("CUDA", "available" if snapshot.cuda_available else "not available", ok=snapshot.cuda_available))
        self.result_layout.addWidget(self._row(
            "Ollama", f"running, {len(snapshot.installed_models)} model(s) installed" if snapshot.ollama_available else "not running",
            ok=snapshot.ollama_available,
        ))

        if snapshot.errors:
            err = QLabel("\n".join(snapshot.errors))
            err.setStyleSheet("color: #c44; margin-top: 8px; font-size: 12px;")
            err.setWordWrap(True)
            self.result_layout.addWidget(err)

        # Ollama not running is a hard block -- everything downstream needs it.
        # A missing GPU is not: the app still works CPU-only, just slower.
        self.continue_button.setEnabled(snapshot.ollama_available)
        if not snapshot.ollama_available:
            self.continue_button.setText("Install/start Ollama to continue")

    def _on_continue(self) -> None:
        if self.snapshot is not None:
            self.continue_clicked.emit(self.snapshot)
