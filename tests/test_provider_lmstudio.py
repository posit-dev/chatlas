import pytest
from chatlas import ChatLMStudio
from chatlas._provider_lmstudio import has_lmstudio, lmstudio_model_info

from .conftest import assert_tools_simple, assert_turns_existing, assert_turns_system


def lmstudio_chat_fun(**kwargs):
    """Create a ChatLMStudio instance using a locally loaded model."""
    skip_if_no_lmstudio()
    models = lmstudio_model_info("http://localhost:1234")
    if not models:
        pytest.skip("No models loaded in LM Studio")
    # Prefer known-good models; fall back to whatever is loaded
    preferred = ["google/gemma-4-26b-a4b", "zai-org/glm-4.7-flash"]
    model_ids = [m["id"] for m in models]
    candidates = [m for m in preferred if m in model_ids] + model_ids
    return ChatLMStudio(model=candidates[0], **kwargs)


def skip_if_no_lmstudio():
    if not has_lmstudio():
        pytest.skip("LM Studio not found")


def test_lmstudio_simple_request():
    chat = lmstudio_chat_fun(system_prompt="Be as terse as possible; no punctuation")
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert turn.tokens[0] > 0
    assert turn.tokens[1] > 0
    assert turn.finish_reason == "stop"


def test_lmstudio_simple_streaming_request():
    chat = lmstudio_chat_fun(system_prompt="Be as terse as possible; no punctuation")
    res = []
    for chunk in chat.stream("What is 1 + 1?"):
        res.append(chunk)
    assert "2" in "".join(res)


def test_lmstudio_respects_turns_interface():
    assert_turns_system(lmstudio_chat_fun)
    assert_turns_existing(lmstudio_chat_fun)


def test_lmstudio_tool_variations():
    assert_tools_simple(lmstudio_chat_fun)


def test_lmstudio_list_models():
    skip_if_no_lmstudio()
    chat = lmstudio_chat_fun()
    models = chat.list_models()
    assert models is not None
    assert isinstance(models, list)
    assert len(models) > 0
    assert "id" in models[0]


def test_lmstudio_error_no_model():
    skip_if_no_lmstudio()
    models = lmstudio_model_info("http://localhost:1234")
    if not models:
        pytest.skip("No models loaded in LM Studio")
    model_ids = [m["id"] for m in models]
    with pytest.raises(ValueError, match="Must specify model"):
        ChatLMStudio(model=None)
    # Error message includes the list of available models
    with pytest.raises(ValueError, match=model_ids[0]):
        ChatLMStudio(model=None)


def test_lmstudio_error_model_not_loaded():
    skip_if_no_lmstudio()
    with pytest.raises(ValueError, match="not available in LM Studio"):
        ChatLMStudio(model="this-model-does-not-exist/fake-v1")


def test_lmstudio_no_server():
    with pytest.raises(RuntimeError, match="Can't find locally running LM Studio"):
        ChatLMStudio(
            model="any-model",
            base_url="http://localhost:9999",
        )
