"""Desktop app entry point. Full flow: hardware detection -> model
selection -> placement optimization -> agent.

Run: python -m app.gui.main
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from app.core.hardware import HardwareSnapshot
from app.core.orchestrator import Orchestrator
from app.gui.agent_screen import AgentScreen
from app.gui.model_screen import ModelScreen
from app.gui.placement_screen import PlacementScreen
from app.gui.setup_screen import SetupScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ramer")
        self.resize(720, 720)

        self.orchestrator = Orchestrator()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_screen = SetupScreen()
        self.model_screen = ModelScreen()
        self.placement_screen = PlacementScreen()
        self.agent_screen = AgentScreen(self.orchestrator)

        for screen in (self.setup_screen, self.model_screen, self.placement_screen, self.agent_screen):
            self.stack.addWidget(screen)

        self.setup_screen.continue_clicked.connect(self._go_to_model_screen)
        self.model_screen.model_selected.connect(self._go_to_placement_screen)
        self.placement_screen.optimized.connect(self._go_to_agent_screen)

    def _go_to_model_screen(self, snapshot: HardwareSnapshot) -> None:
        self.orchestrator.hardware = snapshot  # populate the orchestrator with the already-detected snapshot
        self.model_screen.load_for_hardware(snapshot)
        self.stack.setCurrentWidget(self.model_screen)

    def _go_to_placement_screen(self, model: str) -> None:
        self.placement_screen.load_for_model(model)
        self.stack.setCurrentWidget(self.placement_screen)

    def _go_to_agent_screen(self, model: str, num_ctx: int, result) -> None:
        self.agent_screen.configure(model=model, num_gpu=result.num_gpu, num_ctx=num_ctx, baseline_tok_s=result.tok_per_sec)
        self.stack.setCurrentWidget(self.agent_screen)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
