"""CLI entry point. Calls the same Orchestrator the GUI uses -- no business
logic lives here, only argument parsing and console output.

Usage:
    python -m app.cli doctor
    python -m app.cli models
    python -m app.cli benchmark <model> [--num-ctx N] [--force]
    python -m app.cli run <model> --workdir <path> <task...>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.core.orchestrator import Orchestrator
from app.core.proxy_server import DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM


def cmd_doctor(args) -> None:
    orch = Orchestrator()
    snap = orch.detect_hardware()
    print(f"GPU: {snap.gpu.name} ({snap.gpu.total_vram_mib} MiB VRAM)" if snap.gpu.name else "GPU: none detected")
    print(f"System RAM: {snap.system_ram_gib} GB")
    print(f"CPU: {snap.cpu_name}")
    print(f"OS: {snap.os_name}")
    print(f"CUDA available: {snap.cuda_available}")
    print(f"Ollama running: {snap.ollama_available}")
    if snap.installed_models:
        print("Installed models:")
        for m in snap.installed_models:
            print(f"  - {m.name} ({m.parameter_size}, {m.quantization})")
    if snap.errors:
        print("Issues:")
        for e in snap.errors:
            print(f"  ! {e}")


def cmd_models(args) -> None:
    orch = Orchestrator()
    print("Curated models:")
    for m in orch.curated_models():
        print(f"  {m['name']}  {'*' * m['stars']}{'.' * (5 - m['stars'])}  -- {m['note']}")


def cmd_benchmark(args) -> None:
    orch = Orchestrator()
    result = orch.optimize(args.model, num_ctx=args.num_ctx, force=args.force, on_log=print)
    gain = f"  ({result.gain_over_auto_pct:+.1f}% vs auto)" if result.gain_over_auto_pct is not None else ""
    vram = f"  {result.vram_gb} GB VRAM" if result.vram_gb is not None else ""
    print(f"\nrecommended num_gpu={result.num_gpu}  {result.tok_per_sec} tok/s{vram}{gain}")


def cmd_run(args) -> None:
    orch = Orchestrator()
    orch.detect_hardware()
    placement = orch.optimize(args.model, num_ctx=args.num_ctx, on_log=print)
    print(f"[ramer] using num_gpu={placement.num_gpu} ({placement.tok_per_sec} tok/s)\n")

    def permission_callback(tool: str, tool_args: dict) -> bool:
        answer = input(f"\nAgent wants to run {tool}({tool_args}) -- allow? [y/N] ")
        return answer.strip().lower() == "y"

    session = orch.start_agent(
        model=args.model, workspace=Path(args.workdir).resolve(), num_gpu=placement.num_gpu, num_ctx=args.num_ctx,
        permission_callback=permission_callback, baseline_tok_s=placement.tok_per_sec,
    )

    def on_event(event) -> None:
        if event.type == "content" and event.data.get("text"):
            print(event.data["text"])
        elif event.type == "tool_call":
            print(f"[tool] {event.data['name']}({event.data['args']})")
        elif event.type == "tool_result":
            print(f"  -> {str(event.data['result'])[:300]}")
        elif event.type in ("warning", "drift", "stalled"):
            print(f"[{event.type}] {event.data.get('message')}")
        elif event.type == "cancelled":
            print("[cancelled]")

    result = orch.run_agent_task(session, " ".join(args.task), on_event)
    verification = result.get("verification")
    if verification is not None:
        print(f"\nverified: {verification['verified']}")
        if not verification["verified"]:
            print(f"unconfirmed: {verification.get('unconfirmed')}")


def cmd_serve(args) -> None:
    orch = Orchestrator()
    server = orch.start_proxy_server(port=args.port, upstream=args.upstream, on_log=print)
    print(f"[ramer] proxy will listen on http://127.0.0.1:{args.port} -> {args.upstream}")
    print("[ramer] point your agent tool's Ollama base URL here instead of at Ollama directly, e.g.:")
    print(f"        OLLAMA_HOST=http://127.0.0.1:{args.port}")
    print("[ramer] models with no cached placement fall back to Ollama's own 'auto' -- run "
          "`python -m app.cli benchmark <model>` once per model to let Ramer tune it.")
    print("[ramer] Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ramer] stopping...")
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ramer", description="Ramer: hardware-aware local AI coding agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="show detected hardware and Ollama status").set_defaults(func=cmd_doctor)
    sub.add_parser("models", help="list curated models").set_defaults(func=cmd_models)

    p_bench = sub.add_parser("benchmark", help="probe and cache the best placement for a model")
    p_bench.add_argument("model")
    p_bench.add_argument("--num-ctx", type=int, default=4096)
    p_bench.add_argument("--force", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    p_run = sub.add_parser("run", help="run the coding agent on a task")
    p_run.add_argument("model")
    p_run.add_argument("--workdir", required=True)
    p_run.add_argument("--num-ctx", type=int, default=4096)
    p_run.add_argument("task", nargs="+")
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve", help="run the Ramer proxy: an Ollama-API-compatible server your agent tools "
                                            "can point at instead of Ollama directly")
    p_serve.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT)
    p_serve.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="the real Ollama server this proxies to")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
