"""Real end-to-end agent test: a live Ollama model, a real git workspace,
real tool calls. Uses qwen2.5:3b-instruct for speed (100+ tok/s) since this
is exercising the loop/verification machinery, not model quality.
"""
import subprocess

import pytest

from app.core.agent import AgentSession


@pytest.fixture
def git_workspace(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("placeholder")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_agent_completes_a_real_file_creation_task_and_verifies_it(git_workspace):
    session = AgentSession(
        model="qwen2.5:3b-instruct", workspace=git_workspace, num_gpu="auto", num_ctx=4096,
        permission_callback=lambda tool, args: True,
    )
    events = []
    result = session.run(
        "Create a file named hello.txt containing exactly the text 'hello world' (no quotes), then stop.",
        on_event=events.append,
    )
    assert result.get("done") is True
    assert (git_workspace / "hello.txt").exists()

    verification = result["verification"]
    assert verification is not None
    assert verification["verified"] is True
    assert "hello.txt" in verification["files_written"]

    tool_calls = [e for e in events if e.type == "tool_call"]
    assert any(tc.data["name"] == "write_file" for tc in tool_calls)
    assert all(t["tok_s"] >= 0 for t in session.turns)


def test_agent_can_be_cancelled_between_steps(git_workspace):
    session = AgentSession(model="qwen2.5:3b-instruct", workspace=git_workspace, num_gpu="auto", num_ctx=4096)
    session.cancel()  # cancel before it even starts -- simplest deterministic cancellation check
    events = []
    result = session.run("Do anything.", on_event=events.append)
    assert result.get("cancelled") is True
    assert events[0].type == "cancelled"


def test_agent_reports_false_success_correctly_if_it_ever_recurs(git_workspace):
    """Not asserting the model behaves badly -- asserting that IF it claims
    work with zero actual writes, verification catches it. Directly forces
    that condition rather than hoping the model reproduces the rare failure."""
    session = AgentSession(model="qwen2.5:3b-instruct", workspace=git_workspace, num_gpu="auto", num_ctx=4096)
    baseline = None
    # simulate: agent claims done, having written nothing
    verification = session._verify(baseline)
    assert verification["verified"] is True  # nothing claimed, nothing changed -- correctly not flagged as a lie

    session.tool_ctx.files_written.append("phantom.txt")  # claims a write that never happened on disk/git
    verification = session._verify(baseline)
    assert verification["verified"] is False
    assert "phantom.txt" in verification["unconfirmed"]
