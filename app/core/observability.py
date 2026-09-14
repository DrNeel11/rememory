"""Local-only session logging for debugging real user failures. Per the
spec: no user source code or prompts by default -- this logs token counts,
throughput, placement, and verification/drift signals, never message
content or file contents. File *paths* touched are logged (useful for
debugging "why did verification fail"), not their contents.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "app_sessions.jsonl"


def log_session(*, model: str, num_ctx: int, num_gpu, turns: list[dict], result: dict, hardware=None) -> None:
    verification = result.get("verification") or {}
    record = dict(
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model=model, num_ctx=num_ctx, num_gpu=num_gpu,
        steps=len(turns), turns=turns,
        outcome=next((k for k in ("done", "cancelled", "stalled", "max_steps") if result.get(k)), "unknown"),
        verified=verification.get("verified"),
        files_written=verification.get("files_written", []),
        drift_flagged=result.get("drift_flagged", False),
        hardware=dict(
            gpu=hardware.gpu.name, vram_mib=hardware.gpu.total_vram_mib, ram_gib=hardware.system_ram_gib,
        ) if hardware is not None else None,
    )
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_sessions(limit: Optional[int] = None) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    lines = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
    return lines[-limit:] if limit else lines
