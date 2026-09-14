"""Placement engine: wraps scripts/ramer_local.py's already-validated probe/
cache logic behind a callable interface the GUI/orchestrator can use,
instead of re-implementing any of it. Every claim in this module's
docstrings (context-aware caching, headroom-aware tiebreak) is backed by
real experiments recorded in PRODUCT_HYPOTHESES.md -- this file adds no new
placement logic of its own, only a programmatic front door onto it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ramer_local import (  # noqa: E402
    DEFAULT_NUM_CTX, PROBE_PROMPT, cache_key, load_cache, model_digest, probe, save_cache,
)

SUPPORTED_CONTEXTS = [2048, 4096, 8192, 16384]


@dataclass
class PlacementResult:
    num_gpu: object  # int, or "auto"
    tok_per_sec: float
    vram_gb: Optional[float]
    gain_over_auto_pct: Optional[float]
    num_ctx: int
    from_cache: bool


class PlacementEngine:
    def get_cached(self, model: str, num_ctx: int = DEFAULT_NUM_CTX, num_predict: int = 40) -> Optional[dict]:
        digest = model_digest(model)
        key = cache_key(model, digest, num_predict, num_ctx)
        return load_cache().get(key)

    def probe_model(
        self, model: str, num_ctx: int = DEFAULT_NUM_CTX, num_predict: int = 40,
        force: bool = False, on_log: Optional[Callable[[str], None]] = None,
    ) -> PlacementResult:
        """Returns the cached placement if one already exists for this exact
        (model, hardware, context) combination -- the cache key already
        encodes hardware and context, so a hit here is only ever reused
        under the conditions it was actually measured under. Otherwise runs
        a real probe (the num_gpu sweep + headroom-aware tiebreak from
        ramer_local.py), which takes real wall-clock time -- on_log
        surfaces ramer_local.py's own progress lines to the caller (e.g. a
        GUI log panel) instead of running silently."""
        digest = model_digest(model)
        key = cache_key(model, digest, num_predict, num_ctx)
        cache = load_cache()
        if key in cache and not force:
            result = cache[key]
            from_cache = True
        else:
            log = on_log if on_log else (lambda msg: None)
            result = probe(model, num_predict, PROBE_PROMPT, num_ctx=num_ctx, log=log)
            cache[key] = result
            save_cache(cache)
            from_cache = False

        rec = result["recommended"]
        return PlacementResult(
            num_gpu=rec["num_gpu"], tok_per_sec=rec["decode_tok_s"], vram_gb=rec.get("size_vram_gb"),
            gain_over_auto_pct=result.get("gain_over_auto_pct"), num_ctx=num_ctx, from_cache=from_cache,
        )

    def all_cached(self) -> dict:
        return load_cache()
