# Ramer desktop app

The production application, kept separate from `scripts/` (the research
CLI tools and experiments this app is built on top of -- see the repo
root `README.md` and `PRODUCT_HYPOTHESES.md` for that history).
`scripts/ramer_local.py` and `scripts/mini_agent.py` stay as the
validated research prototypes; this app reuses their core logic (GPU
queries, the probe/cache/pick_best algorithm, the streaming chat loop,
the git-verification step) rather than duplicating it.

## Status: Milestones 1-4 of 4 done, plus a proxy server

Hardware detection, Ollama integration, model selection, placement
optimization, and a working coding agent (filesystem/terminal/git tools,
permission prompts, streaming, context tracking, drift detection, git
self-verification, observability logging) are all implemented and
tested against real hardware, a live Ollama server, and real model
calls -- not mocks, not a mockup. Both a GUI and a CLI exist, both
calling the same `Orchestrator`.

**What's genuinely still missing, stated plainly**: MCP tool support
(architecture allows adding it -- see `agent_tools.py`'s tool registry
-- but nothing is wired up), git commit/stage tools (status and diff
only, per spec), automated GUI tests (verification so far is real but
manual -- see below), and any polish beyond "functional." This is an
MVP, not a finished product.

## Run it

```bash
pip install -r app/requirements.txt

# GUI
python -m app.gui.main

# CLI
python -m app.cli doctor
python -m app.cli models
python -m app.cli benchmark qwen2.5:14b-instruct --num-ctx 8192
python -m app.cli run qwen2.5:14b-instruct --workdir D:\some\project "fix the bug in auth.py"

# Proxy server -- point an EXISTING agent tool (Cline, Continue, mini_agent.py,
# anything that already speaks Ollama's API) at this instead of Ollama directly
python -m app.cli serve --port 11435
```

Requires Ollama running locally (`127.0.0.1:11434`) with at least one
model pulled.

## Test it

```bash
python -m pytest app/tests/ -v
```

**51/51 passing.** These are real integration tests against this
machine's actual hardware, a live Ollama server, and (for the agent
tests) real model calls -- no mocks, consistent with this project's
existing rule against faking measurements (the scheduler's pure-ordering
tests are the one exception, by design -- see `test_scheduler.py`).
Running them requires an NVIDIA GPU and at least one Ollama model
installed (`qwen2.5:3b-instruct` and `qwen2.5:14b-instruct`
specifically, since some tests pin those model names for speed/quality
reasons).

## What's actually implemented

**Core** (`app/core/`, no GUI dependency -- this is what the CLI and GUI
both sit on top of via `Orchestrator`):
- `hardware.py` -- real GPU/VRAM/RAM/CPU/OS/Ollama detection. Every field
  is a real measurement or listed in `HardwareSnapshot.errors`.
- `model_runtime.py` -- the `ModelRuntime` abstraction + `OllamaRuntime`,
  a curated 3-model list, VRAM-based recommendation, streaming pull.
- `placement.py` -- wraps `scripts/ramer_local.py`'s `probe()`/cache
  logic (context-aware caching, headroom-aware tiebreak) behind a
  callable `PlacementEngine`, adding no new placement logic of its own.
- `agent_tools.py` -- `read_file`, `write_file` (with the same
  write-shrink guard `mini_agent.py` needed after a real destructive
  failure), `list_directory`, `search_files`, `run_command` (with a
  timeout), `git_status`, `git_diff`. All confined to one `ToolContext`
  workspace (path-traversal refused), all gated through a
  `permission_callback` for the dangerous ones (`write_file`,
  `run_command`). Tool registry is a plain dict so an MCP-provided tool
  can be added later without touching the agent loop.
- `agent.py` -- `AgentSession`, the tool-calling loop over
  `scripts/mini_agent.py`'s streaming `chat()` (parameterized to accept
  this app's own tool schemas rather than duplicated). Tracks context
  usage (warns past 85%), watches for sustained throughput drift against
  a Ramer-measured baseline, supports cooperative cancellation, and ends
  every run with a **git-based self-verification** step -- cross-checks
  every file the agent actually wrote against `git status`, independent
  of whatever the model's own final answer claims. This exists because
  of a real failure during this project's own validation (see
  `PRODUCT_HYPOTHESES.md`): a session that wrote zero files still
  reported "modified 3 files."
- `scheduler.py` -- `RamerScheduler`, the multi-agent chokepoint: model
  residency tracking, priority queueing, sticky same-model affinity, a
  brief keep-hot grace window before paying a model switch. Built and
  validated against a real reproduced multi-agent contention problem, not
  speculatively -- see PRODUCT_HYPOTHESES.md's scheduler-arc sections for
  the full experiment history (Experiment D, the A/B result, the 4-way
  ablation separating what stickiness vs. priority each actually buy).
- `proxy_server.py` -- `RamerProxyServer`: an HTTP server that speaks
  both Ollama's own wire protocol (`/api/chat`, `/api/generate`,
  `/api/tags`, `/api/show`) AND the OpenAI-compatible chat API
  (`/v1/chat/completions`, `/v1/models`), so an existing agent tool can
  point at Ramer instead of Ollama with zero changes on its end --
  whichever surface it already speaks. The tool-calling paths on both
  surfaces are routed through `RamerScheduler`, with per-model placement
  resolved from `PlacementEngine`'s cache instead of Ollama's own
  conservative "auto". This is the actual shape of the product thesis
  worked out in PRODUCT_HYPOTHESES.md's product-vision section: Ramer as
  invisible infrastructure underneath whatever agent tool a developer
  already uses, not a separate app they have to learn. The OpenAI-compat
  surface exists because a real tool (the `pi` CLI) needed it -- Ollama's
  native and OpenAI-compatible tool-call formats differ in one real way
  (parsed-object vs. JSON-string arguments), handled by
  `_normalize_openai_messages`/`_ollama_tool_calls_to_openai`.
- `observability.py` -- logs model/context/placement/tok-s/verification/
  drift per session to `app_sessions.jsonl`, structurally excluding
  message content and task text (checked by a test, not just by
  intention) -- no source code or prompts collected, per spec.
- `orchestrator.py` -- the one object the GUI and CLI both depend on.

**GUI** (`app/gui/`, PySide6): Setup -> Model -> Placement -> Agent, each
screen backed by a background `QThread` so hardware detection, model
pulls, placement probes, and agent turns never block the UI. Dangerous
tool calls raise a real modal dialog via a `PermissionBridge` that
blocks the worker thread (not the UI thread) until the user answers --
the one genuinely tricky piece of Qt plumbing here, documented in its
own file.

**CLI** (`app/cli.py`): `doctor`, `models`, `benchmark`, `run` -- all
argument-parsing and printing only, calling the same `Orchestrator`
methods the GUI calls. Verified for real: `run` completed a real
file-creation task end-to-end with a piped permission approval.

## What was tested vs. what's a smoke test

**Automated (`pytest`, 38 tests, all passing, no mocks)**: hardware
detection, Ollama integration, placement cache (including one real
fresh probe, not just cache hits), agent tools (workspace boundary,
write-shrink guard, permission gating, real git status/diff against a
real repo), the full agent loop end-to-end (a real model call creating
and verifying a real file, plus cancellation and a forced-mismatch
verification check), the orchestrator (including confirming the
observability log excludes message content), and CLI smoke tests.

**Manual, real, but not automated**: the full GUI flow (Setup -> Model
-> Placement -> Agent) driven by simulated clicks with each screen's
pixmap grabbed directly from the Qt process (never an OS screenshot,
which would capture whatever else is on the user's actual screen) --
confirmed real hardware, real cached placement, and a real agent task
(`list_directory` on a real temp git repo) all render and complete
correctly through the full click-path. The permission dialog itself
(`PermissionBridge`) was exercised via the CLI's piped-input path, not
yet via an automated GUI click on the actual `QMessageBox` -- worth
building out with `pytest-qt` before this goes past MVP.

## Known compromises, stated rather than hidden

- The Setup and Model screens (Milestone 1) call `detect_hardware()` /
  `OllamaRuntime()` directly rather than through `Orchestrator`, which
  the Placement and Agent screens (Milestones 2-4) do use. Not a
  correctness bug -- both paths go through the same underlying
  `ModelRuntime` -- but it's an inconsistency worth cleaning up rather
  than a deliberate design choice.
- `search_files` is filename-glob only, not full-text search.
- The agent's drift watch compares each full model-call's tok/s against
  a baseline, not a mid-generation rolling token window like
  `ramer_local.py`'s `run_with_drift_watch` -- a reasonable adaptation
  for a multi-turn tool-calling loop (there's no single long generation
  to watch a window within, most turns are short), but a real
  simplification worth being explicit about.
- No MCP tools, no git commit/stage, no cloud routing, no accounts --
  all correctly out of scope per the spec, not overlooked.
