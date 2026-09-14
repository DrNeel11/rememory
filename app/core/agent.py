"""The agent loop: reuses scripts/mini_agent.py's validated streaming
chat() (see its docstring for why streaming + a bounded per-token timeout
matter) with this app's own ToolContext-based tools instead of mini_agent's
global-SANDBOX ones.

Also carries forward mini_agent.py's own real-usage findings, not just its
code: a rolling-window drift watch (the same DRIFT_WINDOW/WARMUP/SUSTAIN
shape ramer_local.py's run_with_drift_watch validated) and the git-based
self-verification step (the fix for the run that reported "modified 3
files" while actually writing zero -- see PRODUCT_HYPOTHESES.md). Neither
is optional polish; both came out of a real failure.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from mini_agent import chat, git_changed_files  # noqa: E402 -- reuse, don't duplicate

from app.core.agent_tools import TOOL_SCHEMAS, ToolContext, call_tool

MAX_STEPS = 15
CONTEXT_WARN_RATIO = 0.85

DRIFT_WINDOW = 8
DRIFT_WARMUP = 3  # step-level warmup (a chat call, not a token window) -- ignore early noisy calls
DRIFT_SUSTAIN = 3
QUICK_CHECK_REGRESSION_TOL = 0.7

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful coding agent working inside one project directory. Use the "
    "provided tools to read and modify files, run commands, and inspect git state. "
    "Take small, verifiable steps. When you are done, give a plain-text summary of "
    "exactly what you changed. Always respond in English."
)


@dataclass
class AgentEvent:
    type: str  # "content" | "tool_call" | "tool_result" | "warning" | "drift" | "done" | "cancelled" | "stalled"
    data: dict = field(default_factory=dict)


@dataclass
class AgentSession:
    model: str
    workspace: Path
    num_gpu: object  # int or "auto"
    num_ctx: int
    permission_callback: Optional[Callable[[str, dict], bool]] = None
    baseline_tok_s: Optional[float] = None  # from a Ramer placement probe, if available -- used for drift detection
    chat_fn: Callable = chat  # override to route through a scheduler instead of calling Ollama directly (see scripts/exp_multiagent_scheduler.py)

    def __post_init__(self):
        self.tool_ctx = ToolContext(workspace=Path(self.workspace), permission_callback=self.permission_callback)
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self._cancelled = False
        self.turns: list[dict] = []

    def cancel(self) -> None:
        """Cooperative cancellation, checked between steps and between tool
        calls within a step -- not a hard kill mid-generation, since Ollama
        itself has no cancel-in-flight endpoint this app calls."""
        self._cancelled = True

    def run(self, task: str, on_event: Callable[[AgentEvent], None]) -> dict:
        self.messages.append({"role": "user", "content": task})
        baseline_changed = git_changed_files(self.tool_ctx.workspace)
        low_windows_in_a_row = 0

        for step in range(1, MAX_STEPS + 1):
            if self._cancelled:
                on_event(AgentEvent("cancelled", {}))
                return {"cancelled": True}

            try:
                resp = self.chat_fn(self.model, self.messages, self.num_gpu, self.num_ctx, tools=TOOL_SCHEMAS)
            except (TimeoutError, OSError) as e:
                on_event(AgentEvent("stalled", {"message": str(e)}))
                return {"stalled": True}

            eval_count = resp.get("eval_count") or 0
            eval_duration_ns = resp.get("eval_duration") or 1
            tok_s = round(eval_count / (eval_duration_ns / 1e9), 2) if eval_count else 0.0
            self.turns.append(dict(step=step, prompt_tokens=resp.get("prompt_eval_count", 0), gen_tokens=eval_count, tok_s=tok_s))

            if self.baseline_tok_s and self.baseline_tok_s > 0 and step > DRIFT_WARMUP:
                if tok_s < self.baseline_tok_s * QUICK_CHECK_REGRESSION_TOL:
                    low_windows_in_a_row += 1
                    if low_windows_in_a_row >= DRIFT_SUSTAIN:
                        on_event(AgentEvent("drift", {
                            "message": f"throughput sustained below baseline ({tok_s} vs {self.baseline_tok_s} tok/s "
                                       f"expected) for {low_windows_in_a_row} calls -- placement may need re-probing.",
                        }))
                else:
                    low_windows_in_a_row = 0

            msg = resp.get("message", {})
            self.messages.append(msg)
            on_event(AgentEvent("content", {"text": msg.get("content", ""), "tok_s": tok_s}))

            used = resp.get("prompt_eval_count") or 0
            if used > CONTEXT_WARN_RATIO * self.num_ctx:
                on_event(AgentEvent("warning", {
                    "message": f"context at {used}/{self.num_ctx} tokens -- close to the limit where Ollama "
                               f"silently drops the oldest messages.",
                }))

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                verification = self._verify(baseline_changed)
                on_event(AgentEvent("done", {"content": msg.get("content", ""), "verification": verification, "turns": self.turns}))
                return {"done": True, "verification": verification}

            for call in tool_calls:
                if self._cancelled:
                    on_event(AgentEvent("cancelled", {}))
                    return {"cancelled": True}
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                on_event(AgentEvent("tool_call", {"name": name, "args": args}))
                result = call_tool(self.tool_ctx, name, args)
                on_event(AgentEvent("tool_result", {"name": name, "result": result}))
                self.messages.append({"role": "tool", "content": result})

        on_event(AgentEvent("warning", {"message": "stopped: reached the step limit without a final answer"}))
        return {"max_steps": True}

    def _verify(self, baseline_changed: Optional[list]) -> Optional[dict]:
        """Ground truth from git, not from the model's own summary -- see
        mini_agent.py's finish() for the real failure (a session that wrote
        zero files but reported three "modified") this is copied from."""
        final_changed = git_changed_files(self.tool_ctx.workspace)
        if final_changed is None:
            return None  # not a git repo -- nothing to verify against
        new_changes = [f for f in final_changed if f not in (baseline_changed or [])]
        written = self.tool_ctx.files_written
        if not written:
            return {"verified": not new_changes, "files_written": [], "git_changes": new_changes}
        unconfirmed = [p for p in written if p not in new_changes]
        return {"verified": not unconfirmed, "files_written": written, "git_changes": new_changes, "unconfirmed": unconfirmed}
