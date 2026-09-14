"""Real tests against the actual placement cache and, in one case, a real
fresh probe -- same no-mocks rule as the other app/tests modules.
"""
from app.core.placement import PlacementEngine, SUPPORTED_CONTEXTS


def test_get_cached_returns_a_real_prior_result():
    eng = PlacementEngine()
    cached = eng.get_cached("qwen2.5:14b-instruct", num_ctx=8192)
    assert cached is not None, "expected a prior probe at ctx=8192 on this dev machine"
    assert "recommended" in cached


def test_probe_model_hits_cache_without_reprobing():
    eng = PlacementEngine()
    result = eng.probe_model("qwen2.5:14b-instruct", num_ctx=8192)
    assert result.from_cache is True
    assert isinstance(result.num_gpu, int)
    assert result.tok_per_sec > 0


def test_all_cached_has_multiple_real_entries():
    eng = PlacementEngine()
    cache = eng.all_cached()
    assert len(cache) >= 5


def test_supported_contexts_is_a_small_real_list():
    assert SUPPORTED_CONTEXTS == sorted(SUPPORTED_CONTEXTS)
    assert all(c > 0 for c in SUPPORTED_CONTEXTS)


def test_fresh_probe_on_a_fast_model_actually_runs():
    """The one slow, real test: force a fresh probe on the fastest curated
    model (qwen2.5:3b-instruct is 100+ tok/s, so the whole num_gpu sweep
    finishes in well under a minute) to prove probe_model's non-cache path
    genuinely calls ramer_local.py's probe(), not just the cache lookup."""
    eng = PlacementEngine()
    log_lines = []
    result = eng.probe_model("qwen2.5:3b-instruct", num_ctx=4096, force=True, on_log=log_lines.append)
    assert result.from_cache is False
    assert result.tok_per_sec > 0
    assert len(log_lines) > 3, "expected ramer_local.py's real progress lines to come through on_log"
