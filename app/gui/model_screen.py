from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar

from app.core.hardware import HardwareSnapshot
from app.core.model_runtime import CURATED_MODELS, OllamaRuntime, recommend_model
from app.gui.workers import ModelPullWorker


class ModelCard(QFrame):
    download_clicked = Signal(str)
    use_clicked = Signal(str)

    def __init__(self, name: str, stars: int, note: str, installed: bool, recommended: bool, parent=None):
        super().__init__(parent)
        self.name = name
        self.setStyleSheet("QFrame { border: 1px solid #333; border-radius: 6px; padding: 8px; margin-bottom: 6px; }")
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel(f"{name}  {'★' * stars}{'☆' * (5 - stars)}")
        title.setStyleSheet("font-weight: 600;")
        header.addWidget(title)
        if recommended:
            tag = QLabel("Recommended for your computer")
            tag.setStyleSheet("color: #4a9; font-size: 11px;")
            header.addWidget(tag)
        header.addStretch()
        layout.addLayout(header)

        note_label = QLabel(note)
        note_label.setStyleSheet("color: #888; font-size: 12px;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.action_button = QPushButton()
        self.action_button.clicked.connect(lambda: self.download_clicked.emit(self.name))
        layout.addWidget(self.action_button)

        self.use_button = QPushButton("Use this model")
        self.use_button.clicked.connect(lambda: self.use_clicked.emit(self.name))
        layout.addWidget(self.use_button)

        self._set_installed(installed)

    def _set_installed(self, installed: bool) -> None:
        if installed:
            self.action_button.setText("Installed ✓")
            self.action_button.setEnabled(False)
        else:
            self.action_button.setText("Download")
            self.action_button.setEnabled(True)
        self.use_button.setEnabled(installed)

    def show_progress(self, chunk: dict) -> None:
        self.progress.setVisible(True)
        self.action_button.setEnabled(False)
        total = chunk.get("total")
        completed = chunk.get("completed")
        status = chunk.get("status", "")
        if total and completed is not None:
            pct = int(100 * completed / total)
            self.progress.setValue(pct)
            self.status_label.setText(f"{status} ({pct}%)")
        else:
            self.status_label.setText(status)

    def finish(self, ok: bool) -> None:
        self.progress.setVisible(False)
        if ok:
            self.status_label.setText("")
            self._set_installed(True)
        else:
            self.status_label.setText("Download failed -- check Ollama is running and try again.")
            self.action_button.setEnabled(True)


class ModelScreen(QWidget):
    model_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Choose a model")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        self.layout_.addWidget(title)

        self.cards_container = QVBoxLayout()
        self.layout_.addLayout(self.cards_container)
        self.layout_.addStretch()

        self._runtime = OllamaRuntime()
        self._workers: dict[str, ModelPullWorker] = {}  # keep references alive while a pull runs
        self._cards: dict[str, ModelCard] = {}

    def load_for_hardware(self, snapshot: HardwareSnapshot) -> None:
        while self.cards_container.count():
            item = self.cards_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        recommended = recommend_model(snapshot.gpu.total_vram_mib)
        installed_names = {m.name for m in snapshot.installed_models}

        for entry in CURATED_MODELS:
            card = ModelCard(
                name=entry["name"], stars=entry["stars"], note=entry["note"],
                installed=entry["name"] in installed_names, recommended=(entry["name"] == recommended),
            )
            card.download_clicked.connect(self._start_download)
            card.use_clicked.connect(self.model_selected.emit)
            self.cards_container.addWidget(card)
            self._cards[entry["name"]] = card

    def _start_download(self, model: str) -> None:
        card = self._cards[model]
        worker = ModelPullWorker(model)
        worker.progress.connect(card.show_progress)
        worker.finished_ok.connect(lambda ok: self._on_pull_finished(model, ok))
        self._workers[model] = worker
        worker.start()

    def _on_pull_finished(self, model: str, ok: bool) -> None:
        self._cards[model].finish(ok)
        self._workers.pop(model, None)
