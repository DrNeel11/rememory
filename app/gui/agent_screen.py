from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QListWidget, QTextEdit,
    QLineEdit, QFileDialog, QFrame,
)

from app.core.orchestrator import Orchestrator
from app.gui.permission_bridge import PermissionBridge
from app.gui.workers import AgentWorker

PROJECTS_FILE = Path.home() / ".ramer_app_projects.json"


def _load_projects() -> list[str]:
    if PROJECTS_FILE.exists():
        try:
            return json.loads(PROJECTS_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []


def _save_projects(projects: list[str]) -> None:
    PROJECTS_FILE.write_text(json.dumps(projects))


class AgentScreen(QWidget):
    def __init__(self, orchestrator: Orchestrator, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.model: str | None = None
        self.num_gpu = "auto"
        self.num_ctx = 4096
        self.baseline_tok_s: float | None = None
        self.workspace: Path | None = None
        self.permission_bridge = PermissionBridge()
        self._worker: AgentWorker | None = None

        root = QHBoxLayout(self)

        # Sidebar: projects
        sidebar = QVBoxLayout()
        sidebar.addWidget(QLabel("Projects"))
        self.project_list = QListWidget()
        for p in _load_projects():
            self.project_list.addItem(p)
        self.project_list.itemClicked.connect(self._on_project_selected)
        sidebar.addWidget(self.project_list)
        add_button = QPushButton("+ Add project")
        add_button.clicked.connect(self._add_project)
        sidebar.addWidget(add_button)
        sidebar_frame = QFrame()
        sidebar_frame.setLayout(sidebar)
        sidebar_frame.setFixedWidth(200)
        root.addWidget(sidebar_frame)

        # Main column: status bar, chat, input, stop
        main_col = QVBoxLayout()

        self.status_label = QLabel("No project selected.")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        main_col.addWidget(self.status_label)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        main_col.addWidget(self.chat_view)

        input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Describe what you want the agent to do...")
        self.input_box.returnPressed.connect(self._send_task)
        input_row.addWidget(self.input_box)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_task)
        input_row.addWidget(self.send_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_agent)
        input_row.addWidget(self.stop_button)
        main_col.addLayout(input_row)

        root.addLayout(main_col)

    def configure(self, model: str, num_gpu, num_ctx: int, baseline_tok_s: float | None) -> None:
        self.model = model
        self.num_gpu = num_gpu
        self.num_ctx = num_ctx
        self.baseline_tok_s = baseline_tok_s
        self._update_status()

    def _update_status(self, speed: float | None = None) -> None:
        gpu_name = self.orchestrator.hardware.gpu.name if self.orchestrator.hardware else "unknown"
        parts = [
            f"Model: {self.model or '-'}", f"Context: {self.num_ctx}", f"GPU: {gpu_name}",
            f"Placement: {self.num_gpu}",
        ]
        if speed is not None:
            parts.append(f"Speed: {speed} tok/s")
        parts.append(f"Project: {self.workspace or '(none)'}")
        self.status_label.setText("  |  ".join(parts))

    def _add_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a project folder")
        if not path:
            return
        projects = _load_projects()
        if path not in projects:
            projects.append(path)
            _save_projects(projects)
        self.project_list.addItem(path)
        self._set_workspace(path)

    def _on_project_selected(self, item) -> None:
        self._set_workspace(item.text())

    def _set_workspace(self, path: str) -> None:
        self.workspace = Path(path)
        self._update_status()
        self.chat_view.append(f"<i>Current project: {path}</i>")

    def _permission_callback(self, tool: str, args: dict) -> bool:
        return self.permission_bridge.ask(tool, args)

    def _send_task(self) -> None:
        task = self.input_box.text().strip()
        if not task or not self.workspace or not self.model:
            return
        self.input_box.clear()
        self.chat_view.append(f"<b>You:</b> {task}")
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self._worker = AgentWorker(
            model=self.model, workspace=self.workspace, num_gpu=self.num_gpu, num_ctx=self.num_ctx, task=task,
            permission_callback=self._permission_callback, baseline_tok_s=self.baseline_tok_s,
        )
        self._worker.event.connect(self._on_agent_event)
        self._worker.finished_ok.connect(self._on_agent_finished)
        self._worker.start()

    def _stop_agent(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_agent_event(self, event) -> None:
        if event.type == "tool_call":
            self.chat_view.append(f"<i>Agent: {event.data['name']}({event.data['args']})</i>")
        elif event.type == "tool_result":
            preview = str(event.data["result"])[:200]
            self.chat_view.append(f"<span style='color:#888'>  -&gt; {preview}</span>")
        elif event.type == "content" and event.data.get("text"):
            self.chat_view.append(f"<b>Agent:</b> {event.data['text']}")
            self._update_status(speed=event.data.get("tok_s"))
        elif event.type == "warning":
            self.chat_view.append(f"<span style='color:#c80'>Warning: {event.data['message']}</span>")
        elif event.type == "drift":
            self.chat_view.append(f"<span style='color:#c44'>Drift: {event.data['message']}</span>")
        elif event.type == "stalled":
            self.chat_view.append(f"<span style='color:#c44'>Stalled: {event.data['message']}</span>")
        elif event.type == "cancelled":
            self.chat_view.append("<i>Stopped by user.</i>")

    def _on_agent_finished(self, result: dict) -> None:
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        verification = result.get("verification")
        if verification is not None and not verification.get("verified", True):
            self.chat_view.append(
                f"<span style='color:#c44'><b>Unverified:</b> the agent claimed changes to "
                f"{verification.get('unconfirmed')} but git shows no such change. Check `git diff` yourself.</span>"
            )
