from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QPlainTextEdit,
)

from app.core.placement import PlacementResult, SUPPORTED_CONTEXTS
from app.gui.workers import PlacementWorker


class PlacementScreen(QWidget):
    optimized = Signal(str, int, object)  # model, num_ctx, PlacementResult

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model: str | None = None
        self._worker: PlacementWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Optimize for your hardware")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        self.model_label = QLabel()
        self.model_label.setStyleSheet("color: #888;")
        layout.addWidget(self.model_label)

        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel("Context length"))
        self.ctx_combo = QComboBox()
        for c in SUPPORTED_CONTEXTS:
            self.ctx_combo.addItem(str(c), c)
        self.ctx_combo.setCurrentIndex(1)  # 4096 default, matches Ollama's own default
        ctx_row.addWidget(self.ctx_combo)
        ctx_row.addStretch()
        layout.addLayout(ctx_row)

        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self._start_optimize)
        layout.addWidget(self.optimize_button)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Optimization progress will appear here...")
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.log_view)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.continue_button = QPushButton("Continue")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self._on_continue)
        layout.addWidget(self.continue_button)

        self._last_result: PlacementResult | None = None

    def load_for_model(self, model: str) -> None:
        self.model = model
        self.model_label.setText(f"Model: {model}")
        self.log_view.clear()
        self.result_label.setText("")
        self.continue_button.setEnabled(False)

    def _start_optimize(self) -> None:
        if not self.model:
            return
        num_ctx = self.ctx_combo.currentData()
        self.optimize_button.setEnabled(False)
        self.optimize_button.setText("Optimizing...")
        self.log_view.clear()
        self._worker = PlacementWorker(self.model, num_ctx)
        self._worker.log_line.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_finished(self, result: PlacementResult) -> None:
        self._last_result = result
        self.optimize_button.setEnabled(True)
        self.optimize_button.setText("Re-optimize")

        source = "cached result" if result.from_cache else "fresh probe"
        gain = f", {result.gain_over_auto_pct:+.1f}% vs Ollama's auto placement" if result.gain_over_auto_pct is not None else ""
        vram = f", {result.vram_gb:.2f} GB VRAM" if result.vram_gb is not None else ""
        self.result_label.setText(
            f"Placement: num_gpu={result.num_gpu} at {result.tok_per_sec} tok/s{vram}{gain} ({source})"
        )
        self.continue_button.setEnabled(True)

    def _on_continue(self) -> None:
        if self.model and self._last_result:
            num_ctx = self.ctx_combo.currentData()
            self.optimized.emit(self.model, num_ctx, self._last_result)
