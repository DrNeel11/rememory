"""Real filesystem/git operations against a throwaway tmp_path workspace --
no mocks needed here since these are genuinely fast, local, side-effect-only
operations, unlike the hardware/Ollama tests.
"""
import subprocess

import pytest

from app.core.agent_tools import (
    ToolContext, call_tool, tool_list_directory, tool_read_file, tool_run_command, tool_write_file,
)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


def test_write_then_read_round_trips(ctx):
    tool_write_file(ctx, "hello.txt", "hi there")
    assert tool_read_file(ctx, "hello.txt") == "hi there"
    assert "hello.txt" in ctx.files_written


def test_read_missing_file_reports_error_not_exception(ctx):
    assert "does not exist" in tool_read_file(ctx, "nope.txt")


def test_list_directory(ctx):
    tool_write_file(ctx, "a.txt", "1")
    tool_write_file(ctx, "b.txt", "2")
    listing = tool_list_directory(ctx, ".")
    assert "a.txt" in listing and "b.txt" in listing


def test_path_traversal_is_refused(ctx):
    with pytest.raises(ValueError, match="escapes the workspace"):
        ctx.resolve("../../etc/passwd")


def test_write_shrink_guard_refuses_large_overwrite(ctx):
    tool_write_file(ctx, "big.py", "x = 1\n" * 100)  # 700 bytes
    result = tool_write_file(ctx, "big.py", "x = 1")  # 5 bytes -- looks like an accidental gutting
    assert "refusing" in result
    assert tool_read_file(ctx, "big.py") == "x = 1\n" * 100  # original content preserved


def test_write_shrink_guard_allows_a_normal_rewrite(ctx):
    tool_write_file(ctx, "small.py", "x = 1\n" * 100)
    result = tool_write_file(ctx, "small.py", "x = 1\n" * 90)  # a genuine, modest edit
    assert "wrote" in result


def test_run_command_executes_in_workspace(ctx, tmp_path):
    (tmp_path / "marker.txt").write_text("present")
    if subprocess.os.name == "nt":
        out = tool_run_command(ctx, "dir")
    else:
        out = tool_run_command(ctx, "ls")
    assert "marker.txt" in out


def test_run_command_times_out_on_a_hanging_command(ctx, monkeypatch):
    from app.core import agent_tools
    monkeypatch.setattr(agent_tools, "COMMAND_TIMEOUT_S", 1)
    if subprocess.os.name == "nt":
        out = tool_run_command(ctx, "ping -n 10 127.0.0.1 > nul")
    else:
        out = tool_run_command(ctx, "sleep 10")
    assert "timed out" in out


def test_permission_denied_blocks_dangerous_tool(tmp_path):
    ctx_denied = ToolContext(workspace=tmp_path, permission_callback=lambda tool, args: False)
    result = call_tool(ctx_denied, "write_file", {"path": "x.txt", "content": "hi"})
    assert "denied" in result
    assert not (tmp_path / "x.txt").exists()


def test_permission_allowed_lets_dangerous_tool_run(tmp_path):
    ctx_allowed = ToolContext(workspace=tmp_path, permission_callback=lambda tool, args: True)
    result = call_tool(ctx_allowed, "write_file", {"path": "x.txt", "content": "hi"})
    assert "wrote" in result
    assert (tmp_path / "x.txt").exists()


def test_read_only_tools_never_prompt_permission(tmp_path):
    calls = []
    ctx_tracking = ToolContext(workspace=tmp_path, permission_callback=lambda tool, args: calls.append(tool) or True)
    call_tool(ctx_tracking, "list_directory", {"path": "."})
    assert calls == []  # read-only tool should never have hit the permission callback


def test_git_status_and_diff_against_a_real_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("v1")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    ctx_git = ToolContext(workspace=tmp_path)
    assert call_tool(ctx_git, "git_status", {}) == "(clean)"

    (tmp_path / "f.txt").write_text("v2")
    status = call_tool(ctx_git, "git_status", {})
    assert "f.txt" in status
    diff = call_tool(ctx_git, "git_diff", {})
    assert "-v1" in diff and "+v2" in diff


def test_git_tools_on_a_non_git_directory_report_error_not_exception(tmp_path):
    ctx_no_git = ToolContext(workspace=tmp_path)
    assert "error" in call_tool(ctx_no_git, "git_status", {})
