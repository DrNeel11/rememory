"""Smoke tests for the CLI's two non-interactive, fast commands. `benchmark`
and `run` were verified manually (see app/README.md) since they're slow
and/or require piping interactive input -- not worth making every test run
slow for.
"""
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *args], capture_output=True, text=True, cwd="D:/ramer", timeout=30,
    )


def test_doctor_reports_real_hardware():
    result = _run("doctor")
    assert result.returncode == 0
    assert "GPU:" in result.stdout
    assert "Ollama running:" in result.stdout


def test_models_lists_curated_models():
    result = _run("models")
    assert result.returncode == 0
    assert "qwen2.5" in result.stdout
