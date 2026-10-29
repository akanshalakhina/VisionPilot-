import json

import httpx
import pytest

from backend.agent.llm import LLMError, OpenAICompatibleClient, build_llm
from backend.config import Settings


def _client(handler, key="secret-key"):
    return OpenAICompatibleClient("https://example.test/v1/", key, "some-model", transport=httpx.MockTransport(handler))


def test_client_sends_the_openai_format_and_parses_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), auth=request.headers.get("authorization"), body=json.loads(request.content))
        calls = [
            {"id": "a", "type": "function", "function": {"name": "list_ad_variants", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "rank_audience", "arguments": '{"target_fraction": 0.3}'}},
        ]
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": calls}}]})

    tools = [{"type": "function", "function": {"name": "list_ad_variants"}}]
    reply = _client(handler).chat([{"role": "user", "content": "hi"}], tools=tools)

    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "some-model" and seen["body"]["tools"] == tools
    assert [c.name for c in reply.tool_calls] == ["list_ad_variants", "rank_audience"]
    assert reply.tool_calls[1].arguments == {"target_fraction": 0.3}


def test_a_plain_text_reply_has_no_tool_calls_and_no_tools_key():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    reply = _client(handler).chat([{"role": "user", "content": "hi"}])
    assert reply.content == "hello" and reply.tool_calls == []
    assert "tools" not in seen["body"]


def test_unparseable_tool_arguments_are_kept_so_the_tool_can_reject_them():
    def handler(request):
        call = {"id": "a", "function": {"name": "rank_audience", "arguments": "{not json"}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]}}]})

    reply = _client(handler).chat([{"role": "user", "content": "x"}])
    assert "_unparsed_arguments" in reply.tool_calls[0].arguments


def test_http_errors_become_llm_errors_without_leaking_the_key():
    client = _client(lambda request: httpx.Response(401, json={"error": "bad key secret-key"}))
    with pytest.raises(LLMError) as info:
        client.chat([{"role": "user", "content": "x"}])
    assert "401" in str(info.value) and "secret-key" not in str(info.value)


def test_an_unexpected_response_shape_is_an_llm_error():
    client = _client(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}])


def test_the_key_never_appears_in_reprs():
    client = _client(lambda request: httpx.Response(200, json={}))
    assert "secret-key" not in repr(client)
    assert "secret-key" not in repr(Settings(llm_api_key="secret-key"))


def test_build_llm_requires_a_model_and_your_own_key():
    assert build_llm(Settings()) is None
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_llm(Settings(llm_provider="openrouter", llm_api_key="k"))
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        build_llm(Settings(llm_provider="openrouter", llm_model="m"))
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        build_llm(Settings(llm_provider="nonsense", llm_model="m", llm_api_key="k"))
    assert build_llm(Settings(llm_provider="ollama", llm_model="m")).base_url == "http://localhost:11434/v1"
    assert build_llm(Settings(llm_provider="groq", llm_model="m", llm_api_key="k")).model == "m"
