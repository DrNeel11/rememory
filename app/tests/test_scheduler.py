"""Fast, deterministic tests of the scheduler's ordering logic with a fake
chat_impl -- no Ollama needed to verify priority and sticky-affinity
behavior are actually doing what they claim.
"""
import threading
import time

from app.core.scheduler import RamerScheduler


def _fake_chat(model, messages, num_gpu, num_ctx, tools=None, num_predict=None):
    time.sleep(0.05)
    return {"message": {"content": f"ok from {model}"}, "eval_count": 1, "eval_duration": 1}


def test_priority_ordering():
    sched = RamerScheduler(chat_impl=_fake_chat)
    order = []
    original = sched._chat_impl
    sched._chat_impl = lambda model, *a, **k: (order.append(model), original(model, *a, **k))[1]

    # Hold the worker busy on a first request (0.3s) so the low- and
    # high-priority requests below are genuinely queued together when the
    # worker next picks -- avoids the race of a request being dequeued
    # before its "competitor" is even submitted.
    def _slow_chat(model, *a, **k):
        time.sleep(0.3)
        return original(model, *a, **k)
    sched._chat_impl = _slow_chat
    t_blocker = threading.Thread(target=sched.submit, args=("blocker", "modelX", 9, [], "auto", 4096))
    t_blocker.start()
    time.sleep(0.05)  # let the worker actually start on the blocker
    sched._chat_impl = lambda model, *a, **k: (order.append(model), original(model, *a, **k))[1]

    threads = [
        threading.Thread(target=sched.submit, args=("low", "modelZ", 5, [], "auto", 4096)),
        threading.Thread(target=sched.submit, args=("high1", "modelA", 0, [], "auto", 4096)),
        threading.Thread(target=sched.submit, args=("high2", "modelB", 0, [], "auto", 4096)),
    ]
    for t in threads:
        t.start()
    t_blocker.join(timeout=5)
    for t in threads:
        t.join(timeout=5)

    assert order[0] in ("modelA", "modelB"), f"expected a high-priority request first, got {order}"
    assert order[1] in ("modelA", "modelB"), f"expected both high-priority requests before the low one, got {order}"
    sched.stop()


def test_sticky_affinity_prefers_resident_model_over_switching():
    sched = RamerScheduler(chat_impl=_fake_chat)
    # warm up: make modelA resident
    sched.submit("a1", "modelA", priority=0, messages=[], num_gpu="auto", num_ctx=4096)

    order = []
    original = sched._chat_impl
    sched._chat_impl = lambda model, *a, **k: (order.append(model), original(model, *a, **k))[1]

    # both queued at the same priority tier -- modelA (resident) should be
    # picked before modelB (would require a switch), even though modelB's
    # thread may have been submitted first
    t_b = threading.Thread(target=sched.submit, args=("b", "modelB", 1, [], "auto", 4096))
    t_a = threading.Thread(target=sched.submit, args=("a2", "modelA", 1, [], "auto", 4096))
    t_b.start()
    time.sleep(0.01)
    t_a.start()
    t_b.join(timeout=5)
    t_a.join(timeout=5)

    assert order[0] == "modelA", f"expected sticky affinity to prefer the resident model first, got {order}"
    sched.stop()


def test_switch_count_is_tracked():
    sched = RamerScheduler(chat_impl=_fake_chat)
    sched.submit("a", "modelA", 0, [], "auto", 4096)
    sched.submit("b", "modelB", 0, [], "auto", 4096)  # forces a switch
    sched.submit("c", "modelB", 0, [], "auto", 4096)  # same model, no switch
    assert sched.stats["model_switches"] == 1
    assert sched.stats["requests_served"] == 3
    sched.stop()


def test_sticky_false_ignores_resident_model_within_a_priority_tier():
    """The ablation flag: with sticky=False, a same-priority tie should be
    broken by arrival order alone, not by preferring the resident model --
    isolates "priority-only" behavior from "sticky-only" behavior."""
    sched = RamerScheduler(chat_impl=_fake_chat, sticky=False)
    sched.submit("a1", "modelA", priority=0, messages=[], num_gpu="auto", num_ctx=4096)

    order = []
    original = sched._chat_impl
    sched._chat_impl = lambda model, *a, **k: (order.append(model), original(model, *a, **k))[1]

    t_b = threading.Thread(target=sched.submit, args=("b", "modelB", 1, [], "auto", 4096))
    t_a = threading.Thread(target=sched.submit, args=("a2", "modelA", 1, [], "auto", 4096))
    t_b.start()
    time.sleep(0.05)  # modelB is queued well before modelA's follow-up -- arrival order should win
    t_a.start()
    t_b.join(timeout=5)
    t_a.join(timeout=5)

    assert order[0] == "modelB", f"expected arrival order (no sticky preference), got {order}"
    sched.stop()
