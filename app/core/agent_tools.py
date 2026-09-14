"""Agent tools: filesystem, terminal, and git, all confined to one workspace
directory and gated by an injectable permission callback.

Adapted from scripts/mini_agent.py's tool implementations rather than a
straight import: mini_agent.py uses a module-level global SANDBOX because
it's a single-shot CLI script, but this app can have long-lived, possibly
multiple, agent sessions -- an instance-held ToolContext is the right shape
here, not a duplicate of the same idea, a different fit for a different
lifetime. The path-resolution guard and the write-shrink guard are the same
protections mini_agent.py validated for real (including the real failure --
see PRODUCT_HYPOTHESES.md -- that motivated the shrink guard).

Tool registry (TOOL_SCHEMAS + TOOL_FUNCS) is a plain dict on purpose: adding
an MCP-provided tool later means adding entries here, not rewriting the
agent loop that calls them.
"""
from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MAX_TOOL_OUTPUT_CHARS = 4000
COMMAND_TIMEOUT_S = 30
WRITE_SHRINK_GUARD_RATIO = 0.5
WRITE_SHRINK_GUARD_MIN_BYTES = 200

# Tools that mutate the filesystem or run arbitrary commands need explicit
# approval; read-only tools don't prompt on every call, or the agent becomes
# unusable -- this mirrors the spec's own examples (approve `npm install`,
# approve a file write; don't ask permission to read a file or list a dir).
DANGEROUS_TOOLS = {"write_file", "run_command"}


class PermissionDenied(Exception):
    pass


@dataclass
class ToolContext:
    workspace: Path
    permission_callback: Optional[Callable[[str, dict], bool]] = None
    files_written: list = field(default_factory=list)  # ground truth for the agent's own verification step

    def __post_init__(self):
        self.workspace = self.workspace.resolve()

    def resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if self.workspace not in p.parents and p != self.workspace:
            raise ValueError(f"path '{path}' escapes the workspace ({self.workspace})")
        return p

    def check_permission(self, tool: str, args: dict) -> None:
        if tool not in DANGEROUS_TOOLS:
            return
        if self.permission_callback is not None and not self.permission_callback(tool, args):
            raise PermissionDenied(f"user denied permission for {tool}({args})")


def tool_read_file(ctx: ToolContext, path: str) -> str:
    p = ctx.resolve(path)
    if not p.exists():
        return f"error: {path} does not exist"
    return p.read_text(errors="replace")


def tool_write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Full overwrite -- there's no patch/diff tool in V1, same limitation
    mini_agent.py has. The shrink guard is the same fix that came out of a
    real destructive failure there: refuse a same-path write that would
    gut an existing file down to a fraction of its size instead of
    silently applying it."""
    p = ctx.resolve(path)
    if p.exists():
        old_size = p.stat().st_size
        new_size = len(content.encode())
        if old_size >= WRITE_SHRINK_GUARD_MIN_BYTES and new_size < old_size * WRITE_SHRINK_GUARD_RATIO:
            return (f"error: refusing to write {path} -- this would shrink it from {old_size} to {new_size} "
                    f"bytes. Read the full file first and write back the complete content with only the "
                    f"intended change.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    ctx.files_written.append(path)
    return f"wrote {len(content)} bytes to {path}"


def tool_list_directory(ctx: ToolContext, path: str = ".") -> str:
    p = ctx.resolve(path)
    if not p.exists():
        return f"error: {path} does not exist"
    entries = sorted(f"{e.name}/" if e.is_dir() else e.name for e in p.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def tool_search_files(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    """Simple recursive filename glob, not full-text search -- enough for V1
    ("find the test files"), not a grep replacement."""
    root = ctx.resolve(path)
    if not root.exists():
        return f"error: {path} does not exist"
    matches = []
    for p in root.rglob("*"):
        if p.is_file() and fnmatch.fnmatch(p.name, pattern):
            matches.append(str(p.relative_to(ctx.workspace)))
        if len(matches) >= 200:
            break
    return "\n".join(matches) if matches else "(no matches)"


def tool_run_command(ctx: ToolContext, command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=ctx.workspace, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
        )
        out = f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        out = f"error: command timed out after {COMMAND_TIMEOUT_S}s"
    return out if len(out) <= MAX_TOOL_OUTPUT_CHARS else out[:MAX_TOOL_OUTPUT_CHARS] + "... (truncated)"


def tool_git_status(ctx: ToolContext) -> str:
    result = subprocess.run(["git", "-C", str(ctx.workspace), "status", "--short"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return f"error: not a git repository, or git unavailable ({result.stderr.strip()})"
    return result.stdout or "(clean)"


def tool_git_diff(ctx: ToolContext) -> str:
    result = subprocess.run(["git", "-C", str(ctx.workspace), "diff"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return f"error: not a git repository, or git unavailable ({result.stderr.strip()})"
    out = result.stdout or "(no unstaged changes)"
    return out if len(out) <= MAX_TOOL_OUTPUT_CHARS else out[:MAX_TOOL_OUTPUT_CHARS] + "... (truncated)"


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file in the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (overwrite) a text file in the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_directory", "description": "List entries in a workspace directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_files", "description": "Find files by name pattern (glob) under a directory.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the workspace (requires user approval).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "git_status", "description": "Show `git status --short` for the workspace.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "git_diff", "description": "Show unstaged `git diff` for the workspace.",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOL_FUNCS: dict[str, Callable] = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_directory": tool_list_directory,
    "search_files": tool_search_files,
    "run_command": tool_run_command,
    "git_status": tool_git_status,
    "git_diff": tool_git_diff,
}


def call_tool(ctx: ToolContext, name: str, args: dict) -> str:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"error: unknown tool '{name}'"
    try:
        ctx.check_permission(name, args)
    except PermissionDenied as e:
        return f"error: {e}"
    try:
        return func(ctx, **args)
    except Exception as e:
        return f"error: {e}"
