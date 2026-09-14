"""Integration tests against this real machine's hardware and live Ollama
server -- deliberately not mocked. This project's own rule (see
PRODUCT_HYPOTHESES.md) is "don't fake hardware/model measurements," and a
mocked hardware-detection test would just be testing that the mock returns
what the mock was told to return. Requires: an NVIDIA GPU + nvidia-smi, and
Ollama running with at least one model pulled -- true on this dev machine,
and the assumption any contributor's machine running this suite needs.
"""
from app.core import hardware


def test_detect_gpu_returns_real_values():
    gpu = hardware.detect_gpu()
    assert gpu.name is not None, "expected a real NVIDIA GPU on this machine"
    assert gpu.total_vram_mib is not None and gpu.total_vram_mib > 0
    assert gpu.driver_version is not None


def test_detect_system_ram():
    ram = hardware.detect_system_ram_gib()
    assert ram is not None and ram > 0


def test_detect_cpu_name():
    cpu = hardware.detect_cpu_name()
    assert cpu and len(cpu) > 0


def test_detect_os_name():
    os_name = hardware.detect_os_name()
    assert "Windows" in os_name or "Linux" in os_name or "Darwin" in os_name


def test_detect_ollama_and_models():
    available, models = hardware.detect_ollama()
    assert available is True, "expected Ollama to be running for this test"
    assert len(models) > 0, "expected at least one model pulled on this dev machine"
    assert all(m.name for m in models)


def test_full_snapshot_has_no_silent_lies():
    snap = hardware.detect_hardware()
    # every field the UI would show is either real or explicitly flagged in
    # errors -- never a placeholder silently standing in for a real value.
    if snap.gpu.name is None:
        assert any("GPU" in e for e in snap.errors)
    if not snap.ollama_available:
        assert any("Ollama" in e for e in snap.errors)
