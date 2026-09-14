"""Ramer Multi-Agent Scheduler v0 -- the direct response to Experiment D
(see PRODUCT_HYPOTHESES.md): under naive concurrent use, Ollama serializes
requests and thrashes model residency (evict/reload on every switch,
sometimes reloading the same model twice) because agents independently
race to call it with no coordination. This is a single chokepoint all
agent chat() calls go through instead, so residency and ordering are
actual decisions, not an accident of thread scheduling.

Deliberately "nothing fancy," per the explicit v0 scope:
1. Model residency tracking (self._resident_model)
2. An agent queue (self._queue)
3. Model-switch cost measurement (recorded per served request)
4. Simple priority (lower number = served first)
5. A keep-hot-model policy: prefer a same-model request already queued at
   the best priority tier, and briefly wait (KEEP_HOT_GRACE_S) for one to
   arrive before paying a switch, instead of switching away immediately
   just because nothing else happens to be queued yet.

Explicitly NOT attempted here (would be a v1, not this experiment):
per-model placement tuning (that's num_gpu, a different axis, held at
"auto" throughout so this measures scheduling alone), true batching of
multiple requests into one forward pass, or preemption of an in-flight
generation.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.core.hardware import gpu_vram_mib  # noqa: F401 -- available for a future placement-aware v1, unused in v0

KEEP_HOT_GRACE_S = 1.5  # how long to wait for a same-model follow-up before paying a switch to a different model


@dataclass
class ScheduledRequest:
    agent_id: str
    model: str
    priority: int  # lower = served first
    messages: list
    num_gpu: object
    num_ctx: int
    tools: Optional[list]
    num_predict: Optional[int] = None
    on_chunk: Optional[Callable[[dict], None]] = None  # streaming relay hook for the proxy server; unused by chat_impls that don't accept it
    submitted_at: float = field(default_factory=time.time)
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[dict] = None
    error: Optional[Exception] = None


class RamerScheduler:
    """One resident-model chokepoint for every agent's chat() call. Agents
    call submit() (blocking, same call shape as chat()); a single worker
    thread decides ordering and actually talks to Ollama."""

    def __init__(self, chat_impl: Callable, sticky: bool = True):
        self._chat_impl = chat_impl  # the real mini_agent.chat, injected so this stays testable without Ollama
        self.sticky = sticky  # False isolates priority-only behavior for the FIFO/sticky/priority ablation
        self._queue: list[ScheduledRequest] = []
        self._lock = threading.Lock()
        self._new_request = threading.Event()
        self._resident_model: Optional[str] = None
        self._stop = threading.Event()
        self.stats = dict(model_switches=0, requests_served=0, serve_log=[])
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._new_request.set()

    def submit(self, agent_id: str, model: str, priority: int, messages: list, num_gpu, num_ctx: int,
               tools: Optional[list] = None, num_predict: Optional[int] = None,
               on_chunk: Optional[Callable[[dict], None]] = None) -> dict:
        req = ScheduledRequest(agent_id=agent_id, model=model, priority=priority, messages=messages,
                                num_gpu=num_gpu, num_ctx=num_ctx, tools=tools, num_predict=num_predict,
                                on_chunk=on_chunk)
        with self._lock:
            self._queue.append(req)
        self._new_request.set()
        req.done.wait()
        if req.error is not None:
            raise req.error
        return req.result

    def _pick_next(self) -> Optional[ScheduledRequest]:
        with self._lock:
            if not self._queue:
                return None
            best_priority = min(r.priority for r in self._queue)
            candidates = [r for r in self._queue if r.priority == best_priority]
            if self.sticky:
                same_model = [r for r in candidates if r.model == self._resident_model]
                chosen = same_model[0] if same_model else candidates[0]
            else:
                chosen = candidates[0]  # priority-only: no same-model preference within a tier
            self._queue.remove(chosen)
            return chosen

    def _run(self) -> None:
        while not self._stop.is_set():
            req = self._pick_next()
            if req is None:
                self._new_request.wait(timeout=0.2)
                self._new_request.clear()
                continue

            # the very first request ever has nothing to switch away from --
            # only count/wait-for a switch once something is actually resident
            is_switch = self._resident_model is not None and req.model != self._resident_model
            if is_switch and self.sticky:
                # keep-hot grace: give a same-model follow-up (e.g. this
                # agent's next turn, or another agent already on this
                # model) a short window to show up before paying the
                # switch -- the direct fix for Experiment D's observed
                # double-reload of the same model.
                deadline = time.time() + KEEP_HOT_GRACE_S
                while time.time() < deadline:
                    with self._lock:
                        same_model_waiting = any(r.model == self._resident_model for r in self._queue)
                    if same_model_waiting:
                        req2 = self._pick_next()
                        if req2 is not None:
                            with self._lock:
                                self._queue.append(req)  # put the switch-requiring one back, serve the sticky one first
                            req = req2
                            is_switch = False
                        break
                    time.sleep(0.1)

            t0 = time.time()
            try:
                extra = {"on_chunk": req.on_chunk} if req.on_chunk is not None else {}
                resp = self._chat_impl(req.model, req.messages, req.num_gpu, req.num_ctx,
                                        tools=req.tools, num_predict=req.num_predict, **extra)
                req.result = resp
            except Exception as e:
                req.error = e
            serve_s = time.time() - t0

            if is_switch:
                self.stats["model_switches"] += 1
            self.stats["requests_served"] += 1
            self.stats["serve_log"].append(dict(
                agent=req.agent_id, model=req.model, switch=is_switch,
                wait_s=round(t0 - req.submitted_at, 2), serve_s=round(serve_s, 2),
            ))
            self._resident_model = req.model
            req.done.set()
