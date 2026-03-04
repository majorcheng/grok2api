import asyncio
import json

import orjson
import pytest

from app.api.v1.chat import ChatCompletionRequest, validate_request
from app.core.exceptions import ValidationException
from app.services.grok.model import ModelService
from app.services.grok.processor import CollectProcessor, extract_tool_calls_from_text


def _mock_model_valid(monkeypatch):
    monkeypatch.setattr(ModelService, "valid", staticmethod(lambda _model: True))


def _tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_validate_tool_choice_function_exists(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [{"role": "user", "content": "帮我查天气"}],
            "tools": _tool_definitions(),
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )
    validate_request(req)


def test_validate_tool_choice_missing_function_rejected(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [{"role": "user", "content": "帮我查天气"}],
            "tools": _tool_definitions(),
            "tool_choice": {"type": "function", "function": {"name": "missing_tool"}},
        }
    )

    with pytest.raises(ValidationException) as exc:
        validate_request(req)

    assert exc.value.code == "invalid_tool_choice"
    assert exc.value.param == "tool_choice.function.name"


def test_validate_assistant_tool_calls_message(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {"role": "user", "content": "查北京天气"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"},
                        }
                    ],
                },
            ],
            "tools": _tool_definitions(),
        }
    )
    validate_request(req)


def test_validate_user_tool_calls_rejected(monkeypatch):
    _mock_model_valid(monkeypatch)
    req = ChatCompletionRequest.model_validate(
        {
            "model": "grok-4.20-beta",
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "tools": _tool_definitions(),
        }
    )

    with pytest.raises(ValidationException) as exc:
        validate_request(req)

    assert exc.value.code == "invalid_tool_message"
    assert exc.value.param == "messages.0.tool_calls"


def test_extract_tool_calls_from_text_markdown_json():
    content = """```json
{"tool_calls":[{"name":"get_weather","arguments":{"city":"北京"}}]}
```"""
    tool_calls = extract_tool_calls_from_text(content, _tool_definitions())
    assert tool_calls and len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"])["city"] == "北京"


async def _iter_ndjson(items):
    for item in items:
        yield orjson.dumps(item)


def test_collect_processor_returns_tool_calls_finish_reason():
    items = [
        {
            "result": {
                "response": {
                    "modelResponse": {
                        "responseId": "resp_1",
                        "message": '{"tool_calls":[{"name":"get_weather","arguments":{"city":"上海"}}]}',
                    }
                }
            }
        }
    ]

    async def _run():
        processor = CollectProcessor("grok-4.20-beta", tools=_tool_definitions())
        return await processor.process(_iter_ndjson(items))

    result = asyncio.run(_run())
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["city"] == "上海"


def test_collect_processor_falls_back_to_text_for_unknown_tool():
    items = [
        {
            "result": {
                "response": {
                    "modelResponse": {
                        "responseId": "resp_2",
                        "message": '{"tool_calls":[{"name":"unknown_tool","arguments":{"x":1}}]}',
                    }
                }
            }
        }
    ]

    async def _run():
        processor = CollectProcessor("grok-4.20-beta", tools=_tool_definitions())
        return await processor.process(_iter_ndjson(items))

    result = asyncio.run(_run())
    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert isinstance(choice["message"]["content"], str)
    assert "\"unknown_tool\"" in choice["message"]["content"]
