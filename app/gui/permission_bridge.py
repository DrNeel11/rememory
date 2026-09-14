"""Bridges a permission request from the agent's worker thread to a real
Qt dialog on the main thread, and blocks the worker until the user answers.

Why this is needed at all: AgentSession.run() calls the permission_callback
synchronously from inside QThread.run() (a real OS thread, not the GUI
thread) -- but QMessageBox must be created and exec()'d on the main thread.
Signal/slot connections across threads default to queued delivery, which
is exactly the "hand off to the main thread's event loop" behavior needed
here; a threading.Event makes the worker thread actually wait for the
dialog's result instead of racing ahead.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox


class PermissionBridge(QObject):
    _request = Signal(str, dict, object)  # tool, args, (result_holder dict, threading.Event)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._request.connect(self._handle_on_main_thread)

    def ask(self, tool: str, args: dict) -> bool:
        """Called from the worker thread -- blocks until the main thread's
        dialog is dismissed."""
        result_holder: dict = {}
        event = threading.Event()
        self._request.emit(tool, args, (result_holder, event))
        event.wait(timeout=300)  # a human has to be at the keyboard; 5 min is generous, not infinite
        return result_holder.get("allowed", False)

    def _handle_on_main_thread(self, tool: str, args: dict, carrier: tuple) -> None:
        result_holder, event = carrier
        box = QMessageBox()
        box.setWindowTitle("Agent wants to run a command")
        box.setText(f"Agent wants to run:\n\n{tool}({args})")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        choice = box.exec()
        result_holder["allowed"] = choice == QMessageBox.StandardButton.Yes
        event.set()
