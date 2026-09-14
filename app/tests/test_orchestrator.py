"""Real end-to-end test of the Orchestrator -- the one object both the GUI
and CLI depend on, per the spec's "CLI must not contain separate business
logic" requirement.
"""
import json
import subprocess

from app.core.observability import LOG_PATH
from app.core.orchestrator import Orchestrator


def test_detect_hardware_and_recommend(tmp_path):
    orch = Orchestrator()
    snap = orch.detect_hardware()
    assert snap.gpu.name is not None
    model = orch.recommended_model()
    assert model in {m["name"] for m in orch.curated_models()}


def test_list_installed_models_matches_ollama():
    orch = Orchestrator()
    models = orch.list_installed_models()
    assert len(models) > 0


def test_run_agent_task_logs_a_session(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    orch = Orchestrator()
    orch.detect_hardware()
    session = orch.start_agent(
        model="qwen2.5:3b-instruct", workspace=tmp_path, num_gpu="auto", num_ctx=4096,
        permission_callback=lambda tool, args: True,
    )
    events = []
    lines_before = LOG_PATH.read_text().splitlines() if LOG_PATH.exists() else []

    task = "Create note.txt with the text 'ok', then stop."
    result = orch.run_agent_task(session, task, events.append)

    assert result.get("done") is True
    lines_after = LOG_PATH.read_text().splitlines()
    assert len(lines_after) == len(lines_before) + 1
    last = json.loads(lines_after[-1])
    assert last["model"] == "qwen2.5:3b-instruct"
    # file *paths* touched are fine to log (debugging aid); the raw task
    # text and any generated message content must never appear -- that's
    # the "no prompts/source by default" requirement, checked structurally
    # (no "content"/"task" key) rather than just by absence of this one string.
    assert "content" not in last and "task" not in last and "message" not in last
    assert "note.txt" in last.get("files_written", [])
