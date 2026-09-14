"""Integration tests against the live Ollama server -- same rationale as
test_hardware.py: this project measures real systems, it doesn't mock them.
"""
from app.core.model_runtime import OllamaRuntime, recommend_model, CURATED_MODELS


def test_ollama_available():
    assert OllamaRuntime().available() is True


def test_list_installed_matches_real_server():
    models = OllamaRuntime().list_installed()
    assert len(models) > 0
    assert all(m.installed for m in models)


def test_is_installed_true_for_a_real_model_false_for_a_fake_one():
    rt = OllamaRuntime()
    real_model = rt.list_installed()[0].name
    assert rt.is_installed(real_model) is True
    assert rt.is_installed("definitely-not-a-real-model:latest") is False


def test_get_metadata_returns_block_count_for_an_installed_model():
    rt = OllamaRuntime()
    installed = rt.list_installed()
    assert installed, "need at least one installed model for this test"
    meta = rt.get_metadata(installed[0].name)
    assert meta is not None
    assert meta.block_count is not None and meta.block_count > 0


def test_curated_models_are_small_and_named_plausibly():
    assert 1 <= len(CURATED_MODELS) <= 6, "V1 should stay a small, deliberate list, not a full catalog"
    for m in CURATED_MODELS:
        assert ":" in m["name"]  # Ollama tag shape, e.g. "qwen2.5:7b-instruct"


def test_recommend_model_by_vram():
    assert recommend_model(None) == "qwen2.5:3b-instruct"
    assert recommend_model(4000) == "qwen2.5:3b-instruct"
    assert recommend_model(8000) == "qwen2.5:7b-instruct"
    assert recommend_model(16000) == "qwen2.5:14b-instruct"
