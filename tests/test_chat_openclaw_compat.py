import pytest

from app.api.v1.chat import ChatCompletionRequest, validate_request
from app.core.exceptions import ValidationException
from app.services.grok.model import ModelService


def _mock_model_valid(monkeypatch):
    monkeypatch.setattr(ModelService, "valid", staticmethod(lambda _model: True))


def test_assistant_null_content_is_accepted(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": None},
            ],
        }
    )
    validate_request(req)


def test_tool_null_content_is_accepted(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {"role": "user", "content": "call tool"},
                {"role": "tool", "content": None},
            ],
        }
    )
    validate_request(req)


def test_user_null_content_is_rejected(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {"role": "user", "content": None},
            ],
        }
    )

    with pytest.raises(ValidationException) as exc:
        validate_request(req)

    assert exc.value.code == "empty_content"
    assert exc.value.param == "messages.0.content"


def test_assistant_empty_string_is_rejected(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {"role": "assistant", "content": "   "},
            ],
        }
    )

    with pytest.raises(ValidationException) as exc:
        validate_request(req)

    assert exc.value.code == "empty_content"
    assert exc.value.param == "messages.0.content"


def test_assistant_empty_text_block_is_rejected(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "   "},
                    ],
                },
            ],
        }
    )

    with pytest.raises(ValidationException) as exc:
        validate_request(req)

    assert exc.value.code == "empty_text"
    assert exc.value.param == "messages.0.content.0.text"
