# Ramer

Ramer sits between the local coding/research agent you already use
(Claude Code, Cline, aider, the `pi` CLI, anything that talks to Ollama
natively or via an OpenAI-compatible endpoint) and Ollama itself. You
point your agent tool at Ramer's port instead of Ollama's. Nothing
about your workflow changes. Underneath, Ramer:

- finds the GPU/CPU layer split that actually maximizes throughput for
  each model at the context length you're using (Ollama's own default
  is deliberately conservative and leaves real performance on the
  table -- see "Validated results" below), and
- arbitrates when more than one agent or model wants the same GPU at
  once, instead of letting them independently thrash it.

You don't run a probe or edit a config for each model. You install
Ramer, point your existing tool's base URL at it, and keep working.

## Status

This is an early build going through real-world testing. If you're
reading this because someone pointed you here: that's the point --
unvarnished feedback, including "this is useless for what I actually
do," is exactly what's being asked for. Reply to whoever sent you this
link, or open an issue here.

Every performance claim below is a real measurement on one real
machine (RTX 5060, 8GB VRAM), independently verified where a task has
a checkable outcome (git diff + pytest, not a model's self-report) --
not a benchmark suite and not a general guarantee across hardware.

## Quick start

Requires Python 3.10+, a local Ollama install, and a **discrete NVIDIA
GPU** (`nvidia-smi` on PATH). Built and tested on Windows; the
discrete-GPU assumption should hold on Linux too, but that's untested.
**Not applicable to Apple Silicon Macs** -- this optimizes the VRAM/RAM
split across a PCIe bus, which unified memory doesn't have.

```bash
pip install -r app/requirements.txt

# start the proxy -- an existing agent tool points here instead of at Ollama
python -m app.cli serve --port 11435
```

Then point your agent tool's Ollama (or OpenAI-compatible) base URL at
`http://127.0.0.1:11435` instead of Ollama's own `11434`. That's the
whole integration -- no code changes on the tool's side. Confirmed
working end-to-end with `aider` and with the `pi` CLI, including real
tool-calling agentic tasks, not just chat.

Optional, if you want to see or force a specific model's placement
before using it live:

```bash
python -m app.cli doctor                                    # hardware + Ollama status
python -m app.cli benchmark qwen2.5:14b-instruct --num-ctx 8192
```

A desktop GUI also exists (`python -m app.gui.main`) covering the same
setup flow plus a built-in coding agent, for anyone who'd rather not
use the CLI.

## Validated results

Same methodology throughout: compare against Ollama's own default
placement, then re-run on a real agentic task with the outcome checked
independently, not by trusting the model's or the agent's own report.

| model | Ollama auto | Ramer | gain |
|---|---|---|---|
| Qwen2.5-14B (Q4_K_M) | 12.4 tok/s | 19.6 tok/s | +61% |
| Qwen2.5-32B (Q2_K, VRAM boundary case) | 3.1 tok/s | 3.3 tok/s | +6.7% |
| Qwen3.8-27B (IQ2_XXS) | 5.9 tok/s | 29.6 tok/s | +402% |

And on real end-to-end agent tasks through two different third-party
tools (`aider`, `pi`), not synthetic benchmarks:

| workload | with Ramer | without Ramer | takeaway |
|---|---|---|---|
| coding task, 14B | 55.4s, correct | 89.2s, correct | ~38% faster, same outcome |
| coding task, 32B-Q2_K | fails | fails | placement can't rescue an inadequate model |
| research task (web search + read), 14B | 19.8s median | 19.9s median | no benefit when network/tool latency dominates |

**The honest, bounded thesis this points to**: Ramer's benefit is
proportional to how much of a workload is actual local decoding. It's
a real, repeated win for decode-heavy work (coding, long generations),
negligible for short-answer/tool-latency-bound work, and it does not
make an unreliable model reliable.

## How the placement is kept honest

- **Context-aware**: a placement is only valid at the context window it
  was measured at (the KV cache competes with model weights for the
  same VRAM), so placements are probed and cached per model *and*
  context length.
- **Headroom-aware**: among candidates that tie on throughput, the
  smallest VRAM footprint wins -- a placement that's 0.04 tok/s faster
  but has zero headroom is a worse bet than one with margin to spare.
- **Self-healing**: every run does a cheap pre-flight throughput check
  against the cached number; if it's regressed, that run falls back to
  Ollama's own safe auto-placement and the cache is flagged for a fresh
  probe next time.
- **Scheduled, not just placed**: a single chokepoint (`RamerScheduler`)
  handles priority ordering and sticky same-model residency across
  concurrent agents, so switching between two agents on one GPU doesn't
  thrash.

## What this doesn't do

- **MoE expert-level placement.** Mixture-of-experts models have a
  finer-grained placement lever (keep shared attention on GPU, route
  individual experts to CPU) that `llama.cpp` supports directly but
  Ollama's API doesn't expose. Confirmed by direct measurement; would
  require driving a different backend, not tuning this tool harder.
- **Rescue a model that's genuinely too large or too heavily quantized
  for the task.** Placement optimizes what a model can already do; it
  doesn't add capability. See the 32B-Q2_K rows above.
- **Anything beyond Ollama as the actual inference engine.** The proxy
  speaks Ollama's native API and the OpenAI-compatible chat API on top
  of it, but Ollama itself still does the inference.

## Repo layout

- `app/core/` -- hardware detection, placement engine, multi-agent
  scheduler, the Ollama/OpenAI-compatible proxy server, a coding agent.
- `app/gui/` -- the desktop GUI.
- `app/cli.py` -- the CLI entry point (`serve`, `doctor`, `benchmark`).
- `app/tests/` -- test suite (`pytest app/tests`).

## License

MIT, see `LICENSE`.
