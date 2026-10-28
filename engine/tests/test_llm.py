"""The LLM client: compact schemas, tool selection, constrained steps, the HTTP client and the
cache-friendly conversation.

The client tests run against a scripted OpenAI-compatible server in this process;
``test_real_model`` runs the real llama.cpp and model when ``BRAINO_LLAMA_TESTS=1``.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from braino.evals.tasks import builtin_tasks
from braino.llm import (
    Answer,
    AskUser,
    CallTool,
    ChatClient,
    ContextFullError,
    Conversation,
    LLMError,
    LoadTools,
    RequestDecision,
    StepError,
    catalogue,
    compact_schema,
    parse_step,
    select_tools,
    step_schema,
    system_prompt,
    tool_card,
)
from braino.llm.conversation import MAX_RESULT_CHARS
from braino.llm.schemas import inline_refs, parameters
from braino.tools import registry
from braino.tools.base import Risk, make_tool
from braino.tools.registry import ToolRegistry

# --- compact schemas ---------------------------------------------------------------------


def _instances(schema: dict[str, Any], valid: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """``valid`` and variations of it: each property set to values of every JSON type."""
    yield valid
    yield {}
    for name in schema.get("properties", {}):
        for value in (None, "x", "average", 1, 0.5, -3, True, [], ["P9"], [1, 2], {}, {"a": ["b"]}):
            yield {**valid, name: value}
    yield {**valid, "not_a_parameter": 1}


def test_compact_schemas_allow_exactly_the_same_values() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    examples: dict[str, list[dict[str, Any]]] = {}
    for task in builtin_tasks():
        for step in task.solution:
            if step.tool:
                examples.setdefault(step.tool, []).append(step.args)
    checked = 0
    for tool in registry:
        raw = tool.input_model.model_json_schema()
        compact = parameters(tool)  # what the model is sent
        assert "$ref" not in json.dumps(compact)
        raw_v = jsonschema.Draft202012Validator(raw)
        compact_v = jsonschema.Draft202012Validator(compact)
        for valid in examples.get(tool.name, [{"subject": "01"}]):
            for instance in _instances(raw, valid):
                assert raw_v.is_valid(instance) == compact_v.is_valid(instance), (
                    tool.name,
                    instance,
                )
                checked += 1
    assert checked > 500


def test_compact_schema_drops_titles_and_collapses_optionals() -> None:
    class In(BaseModel):
        title: str = Field(description="A parameter that happens to be called title")
        limit: int | None = None
        mode: str | list[str] = "a"

    compact = compact_schema(In.model_json_schema())
    assert set(compact["properties"]) == {"title", "limit", "mode"}  # the parameter stays
    assert "title" not in compact and "title" not in compact["properties"]["title"]
    assert compact["properties"]["limit"] == {"type": ["integer", "null"], "default": None}
    assert "anyOf" in compact["properties"]["mode"]  # not a plain optional: kept


def test_compact_schemas_are_smaller() -> None:
    raw = sum(len(json.dumps(t.input_model.model_json_schema())) for t in registry)
    compact = sum(
        len(json.dumps(compact_schema(t.input_model.model_json_schema()))) for t in registry
    )
    assert compact < 0.85 * raw


def test_catalogue_and_cards() -> None:
    text = catalogue(list(registry))
    assert len(text.splitlines()) == len(registry)
    assert "- eeg.qc_report: Screen one recording" in text
    assert "eeg.detect_bad_channels:" in text and "[decision: needs the user's approval]" in text
    card = tool_card(registry.get("eeg.ica_clean"))
    assert card["name"] == "eeg.ica_clean"
    assert "approves" in card["description"]
    assert "title" not in card["parameters"]


# --- tool selection ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("Check the data quality of subject 003.", {"eeg.qc_report"}),
        ("Filter subject 01 between 0.1 and 30 Hz.", {"eeg.preprocess"}),
        ("Remove eye blinks with ICA", {"eeg.ica_clean"}),
        ("Compare the P3 between conditions", {"eeg.erp"}),
        ("What is the alpha power at rest?", {"eeg.spectral"}),
        ("Hello", {"dataset.inspect"}),
    ],
)
def test_select_tools(request_text: str, expected: set[str]) -> None:
    chosen = select_tools(request_text, registry)
    assert expected <= set(chosen)
    assert len(chosen) <= 4
    order = [t.name for t in registry]
    assert chosen == sorted(chosen, key=order.index)


def test_select_tools_matches_word_starts_only() -> None:
    assert "eeg.erp" not in select_tools("convert my mp3 files", registry)


# --- steps -------------------------------------------------------------------------------


def _tools(*names: str) -> list[Any]:
    return [registry.get(n) for n in names]


def test_step_schema_lists_only_possible_actions() -> None:
    schema = step_schema(_tools("eeg.qc_report"), ["eeg.epoch"], ["run-3"])
    actions = [o["properties"]["action"]["const"] for o in schema["anyOf"]]
    assert actions == ["call", "call", "load_tools", "decide", "ask", "answer"]
    call = schema["anyOf"][0]
    assert call["properties"]["tool"] == {"const": "eeg.qc_report"}
    assert call["properties"]["arguments"]["required"] == ["subject"]
    assert schema["anyOf"][1]["properties"]["tool"] == {"enum": ["eeg.epoch"]}
    assert schema["anyOf"][2]["properties"]["tools"]["items"] == {"enum": ["eeg.epoch"]}
    assert schema["anyOf"][3]["properties"]["proposal_id"] == {"enum": ["run-3"]}

    minimal = step_schema([], can_ask=False)
    assert [o["properties"]["action"]["const"] for o in minimal["anyOf"]] == ["answer"]


def test_step_schema_is_enforced_by_a_validator() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = step_schema(_tools("eeg.preprocess"), ["eeg.epoch"])
    ok = jsonschema.Draft202012Validator(schema).is_valid
    assert ok({"action": "call", "tool": "eeg.preprocess", "arguments": {"subject": "01"}})
    assert ok({"action": "load_tools", "tools": ["eeg.epoch"]})
    assert ok({"action": "call", "tool": "eeg.epoch", "arguments": {"anything": 1}})
    assert not ok({"action": "call", "tool": "eeg.nope", "arguments": {}})
    assert not ok(
        {
            "action": "call",
            "tool": "eeg.preprocess",
            "arguments": {"subject": "01", "filter_method": "chebyshev"},
        }
    )
    assert not ok({"action": "load_tools", "tools": ["eeg.nope"]})
    assert not ok({"action": "decide", "proposal_id": "x", "message": "m"})
    assert not ok({"action": "answer", "text": ""})


def test_parse_step() -> None:
    loaded = _tools("eeg.qc_report")
    step = parse_step(
        '{"action": "call", "tool": "eeg.qc_report", "arguments": {"subject": "02"}}', loaded
    )
    assert step == CallTool(tool="eeg.qc_report", arguments={"subject": "02"})
    assert parse_step('{"action": "answer", "text": "Done."}', loaded) == Answer(text="Done.")
    assert parse_step('{"action":"ask","question":"Which?"}', loaded) == AskUser(question="Which?")
    assert parse_step(
        '{"action":"decide","proposal_id":"p1","message":"Drop T8?"}', loaded, proposals=["p1"]
    ) == RequestDecision(proposal_id="p1", message="Drop T8?")
    assert parse_step(
        '{"action":"load_tools","tools":["eeg.erp"]}', loaded, ["eeg.erp"]
    ) == LoadTools(tools=["eeg.erp"])

    bad = {
        '{"action": "call", "tool": "eeg.qc_report"': "not JSON",
        '{"action": "dance"}': "not a valid action",
        '{"action": "call", "tool": "eeg.nope", "arguments": {}}': "no tool",
        '{"action": "call", "tool": "eeg.qc_report", "arguments": {}}': "invalid arguments",
        '{"action": "load_tools", "tools": ["eeg.nope"]}': "cannot load",
        '{"action": "decide", "proposal_id": "p9", "message": "?"}': "no pending proposal",
    }
    for content, message in bad.items():
        with pytest.raises(StepError, match=message):
            parse_step(content, loaded, ["eeg.erp"], ["p1"])


# --- a scripted OpenAI-compatible server -------------------------------------------------


class ScriptedServer:
    """Answers ``/v1/chat/completions`` with the queued replies, records every request."""

    def __init__(self) -> None:
        self.replies: list[Any] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.unavailable = 0  # answer 503 this many times first
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                outer.headers.append(dict(self.headers))
                if outer.unavailable:
                    outer.unavailable -= 1
                    self._send(503, {"error": {"message": "Loading model"}})
                elif self.path == "/tokenize":
                    self._send(200, {"tokens": list(range(len(payload["content"].split())))})
                elif self.path == "/v1/chat/completions":
                    reply = outer.replies.pop(0)
                    if isinstance(reply, tuple):
                        self._send(*reply)
                    else:
                        self._send(200, _completion(reply, payload))
                else:
                    self._send(404, {"error": "not found"})

            def _send(self, code: int, body: Any) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def client(self, **kwargs: Any) -> ChatClient:
        return ChatClient(self.url, api_key="secret", retry_pause=0.01, **kwargs)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _completion(reply: Any, payload: dict[str, Any]) -> dict[str, Any]:
    content = reply if isinstance(reply, str) else json.dumps(reply)
    prompt = sum(len(m["content"]) for m in payload["messages"]) // 4
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": len(content) // 4},
        "timings": {"prompt_n": 10, "prompt_ms": 150.0, "predicted_ms": 900.0},
    }


@pytest.fixture
def server() -> Iterator[ScriptedServer]:
    s = ScriptedServer()
    yield s
    s.close()


# --- the client --------------------------------------------------------------------------


def test_client_sends_a_constrained_request(server: ScriptedServer) -> None:
    server.replies.append({"action": "answer", "text": "hi"})
    reply = server.client(model="qwen").chat(
        [{"role": "user", "content": "Hello"}], json_schema={"type": "object"}, max_tokens=64
    )
    payload = server.requests[0]
    assert payload["model"] == "qwen" and payload["max_tokens"] == 64
    assert payload["temperature"] == 0.0
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert payload["cache_prompt"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert server.headers[0]["Authorization"] == "Bearer secret"
    assert json.loads(reply.content) == {"action": "answer", "text": "hi"}
    assert reply.finish_reason == "stop"
    assert reply.usage.prompt_read == 10 and reply.usage.prompt_seconds == 0.15
    assert reply.usage.cached_tokens == reply.usage.prompt_tokens - 10


def test_client_for_other_servers_sends_standard_fields_only(server: ScriptedServer) -> None:
    server.replies.append("plain")
    server.client(llama_cpp=False).chat([{"role": "user", "content": "Hello"}])
    payload = server.requests[0]
    assert "cache_prompt" not in payload and "chat_template_kwargs" not in payload
    assert "response_format" not in payload
    assert server.client(llama_cpp=False).count_tokens("a b c") is None


def test_client_thinking_switch(server: ScriptedServer) -> None:
    server.replies += ["a", "b"]
    client = server.client(thinking=True)
    client.chat([{"role": "user", "content": "x"}])
    client.chat([{"role": "user", "content": "x"}], thinking=False)
    assert [r["chat_template_kwargs"]["enable_thinking"] for r in server.requests] == [True, False]


def test_client_retries_while_the_server_loads(server: ScriptedServer) -> None:
    server.unavailable = 2
    server.replies.append("ok")
    assert server.client().chat([{"role": "user", "content": "x"}]).content == "ok"
    assert len(server.requests) == 3


def test_client_errors(server: ScriptedServer) -> None:
    server.replies.append((400, {"error": {"message": "failed to parse grammar"}}))
    with pytest.raises(LLMError, match=r"HTTP 400.*grammar"):
        server.client().chat([{"role": "user", "content": "x"}])
    server.replies.append((200, {"choices": []}))
    with pytest.raises(LLMError, match="unexpected reply"):
        server.client().chat([{"role": "user", "content": "x"}])
    url = server.url
    server.close()
    with pytest.raises(LLMError, match="could not reach"):
        ChatClient(url, retries=1, retry_pause=0.01).chat([{"role": "user", "content": "x"}])
    server.httpd = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)  # for close()
    threading.Thread(target=server.httpd.serve_forever, daemon=True).start()


def test_count_tokens(server: ScriptedServer) -> None:
    assert server.client().count_tokens("one two three") == 3
    assert server.requests[0] == {"content": "one two three"}


# --- the conversation --------------------------------------------------------------------


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"action": "call", "tool": tool, "arguments": arguments}


def _assert_append_only(requests: list[dict[str, Any]]) -> None:
    for before, after in itertools.pairwise(requests):
        sent = before["messages"]
        assert after["messages"][: len(sent)] == sent, "an earlier message changed"


def test_conversation_preloads_and_only_appends(server: ScriptedServer) -> None:
    server.replies += [
        _call("eeg.qc_report", subject="02"),
        {"action": "answer", "text": "T8 is noisy."},
    ]
    conv = Conversation(server.client(), "Check the data quality of subject 02.")
    first = conv.next_step()
    assert first == CallTool(tool="eeg.qc_report", arguments={"subject": "02"})
    conv.add_result({"noisy_channels": ["T8"]})
    assert conv.next_step() == Answer(text="T8 is noisy.")

    messages = server.requests[0]["messages"]
    assert messages[0] == {"role": "system", "content": system_prompt(registry)}
    assert messages[1] == {"role": "user", "content": "Check the data quality of subject 02."}
    assert json.loads(messages[2]["content"])["action"] == "load_tools"
    assert messages[3]["content"].startswith("<tool_response>\nLoaded.")
    assert '"name": "eeg.qc_report"' in messages[3]["content"]
    _assert_append_only(server.requests)
    last = server.requests[1]["messages"]
    assert last[-1] == {
        "role": "user",
        "content": '<tool_response>\n{"noisy_channels":["T8"]}\n</tool_response>',
    }
    schema = server.requests[0]["response_format"]["json_schema"]["schema"]
    tools = [
        o["properties"]["tool"]["const"]
        for o in schema["anyOf"]
        if "const" in o["properties"].get("tool", {})
    ]
    assert tools == [t.name for t in conv.loaded]


def test_the_system_prompt_is_the_same_for_every_request(server: ScriptedServer) -> None:
    server.replies += [{"action": "answer", "text": "a"}, {"action": "answer", "text": "b"}]
    Conversation(server.client(), "Filter subject 01").next_step()
    Conversation(server.client(), "What alpha power is there?").next_step()
    assert server.requests[0]["messages"][0] == server.requests[1]["messages"][0]


def test_conversation_loads_tools_the_model_asks_for(server: ScriptedServer) -> None:
    server.replies += [
        {"action": "load_tools", "tools": ["eeg.epoch"]},
        _call("eeg.epoch", subject="01", conditions={"a": ["x"]}),
    ]
    conv = Conversation(server.client(), "Hello", preload=[])
    assert conv.loaded == []
    step = conv.next_step()  # the load is handled inside
    assert isinstance(step, CallTool) and step.tool == "eeg.epoch"
    assert [t.name for t in conv.loaded] == ["eeg.epoch"]
    assert isinstance(conv.steps[0], LoadTools)
    assert '"name": "eeg.epoch"' in server.requests[1]["messages"][-1]["content"]
    _assert_append_only(server.requests)


def test_calling_a_tool_that_is_not_loaded_loads_it(server: ScriptedServer) -> None:
    assert parse_step(json.dumps(_call("eeg.erp", x=1)), [], ["eeg.erp"]) == LoadTools(
        tools=["eeg.erp"]
    )
    server.replies += [_call("eeg.erp"), _call("eeg.erp", subject="01")]
    conv = Conversation(server.client(), "Hello", preload=[])
    step = conv.next_step()
    assert step == CallTool(tool="eeg.erp", arguments={"subject": "01"})
    assert [t.name for t in conv.loaded] == ["eeg.erp"]


def test_conversation_repairs_one_invalid_reply(server: ScriptedServer) -> None:
    # A server that does not enforce the schema: first an unknown tool, then a valid answer.
    server.replies += [_call("eeg.nope"), {"action": "answer", "text": "ok"}]
    conv = Conversation(server.client(), "Hello", preload=["dataset.inspect"])
    assert conv.next_step() == Answer(text="ok")
    feedback = server.requests[1]["messages"][-1]["content"]
    assert "Error: there is no tool 'eeg.nope'" in feedback

    server.replies += ["{not json", "{still not"]
    with pytest.raises(StepError, match="not JSON"):
        conv.next_step()


def test_conversation_cut_off_reply(server: ScriptedServer) -> None:
    server.replies += [
        (
            200,
            {
                "choices": [
                    {"message": {"content": '{"action": "answer", "te'}, "finish_reason": "length"}
                ]
            },
        ),
        {"action": "answer", "text": "short"},
    ]
    conv = Conversation(server.client(), "Hello", preload=[])
    assert conv.next_step() == Answer(text="short")
    assert "cut off" in server.requests[1]["messages"][-1]["content"]


def test_conversation_decisions(server: ScriptedServer) -> None:
    server.replies += [{"action": "decide", "proposal_id": "run-2", "message": "Mark T8 bad?"}]
    conv = Conversation(
        server.client(), "Find bad channels in 02", preload=["eeg.detect_bad_channels"]
    )
    conv.add_result({"proposal_id": "run-2", "status": "pending"}, proposal_id="run-2")
    assert conv.pending == ["run-2"]
    step = conv.next_step()
    assert step == RequestDecision(proposal_id="run-2", message="Mark T8 bad?")
    assert conv.pending == []
    schema = server.requests[0]["response_format"]["json_schema"]["schema"]
    decide = next(o for o in schema["anyOf"] if o["properties"]["action"]["const"] == "decide")
    assert decide["properties"]["proposal_id"] == {"enum": ["run-2"]}


def test_inline_refs() -> None:
    schema = registry.get("eeg.erp").input_model.model_json_schema()
    assert "$defs" in schema
    inlined = inline_refs(schema)
    assert "$defs" not in json.dumps(inlined) and "$ref" not in json.dumps(inlined)
    nested = {"type": "object", "properties": {"arguments": inlined}}
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(nested)
    with pytest.raises(ValueError, match="recursive"):
        inline_refs({"$defs": {"A": {"items": {"$ref": "#/$defs/A"}}}, "$ref": "#/$defs/A"})


def test_conversation_loads_the_tools_a_result_names(server: ScriptedServer) -> None:
    server.replies.append(_call("eeg.preprocess", subject="01"))
    conv = Conversation(server.client(), "Epoch subject 01", preload=["eeg.epoch"])
    conv.add_result({"ok": False, "error": "MissingStepError: run eeg.preprocess first"})
    assert [t.name for t in conv.loaded] == ["eeg.epoch", "eeg.preprocess"]
    assert '"name": "eeg.preprocess"' in conv.messages[-1]["content"]
    assert conv.next_step() == CallTool(tool="eeg.preprocess", arguments={"subject": "01"})
    conv.add_result("see eeg.preprocess.apply and eeg.epochs")  # not tool names: nothing loaded
    assert len(conv.loaded) == 2


def test_no_two_questions_in_a_row(server: ScriptedServer) -> None:
    server.replies += [
        {"action": "ask", "question": "Which subject?"},
        {"action": "answer", "text": "ok"},
    ]
    conv = Conversation(server.client(), "Hello", preload=[])
    assert isinstance(conv.next_step(), AskUser)
    conv.add_user_reply("Any.")
    conv.next_step()
    actions = [
        [
            o["properties"]["action"]["const"]
            for o in r["response_format"]["json_schema"]["schema"]["anyOf"]
        ]
        for r in server.requests
    ]
    assert "ask" in actions[0] and "ask" not in actions[1]


def test_conversation_user_replies_and_long_results(server: ScriptedServer) -> None:
    conv = Conversation(server.client(), "Hello", preload=[])
    conv.add_user_reply("Subject 02, please.")
    assert (
        conv.messages[-1]["content"]
        == "<tool_response>\nThe user replied: Subject 02, please.\n</tool_response>"
    )
    conv.add_result("x" * (MAX_RESULT_CHARS + 500))
    assert "(500 more characters not shown)" in conv.messages[-1]["content"]


def test_conversation_stops_before_the_context_is_full(server: ScriptedServer) -> None:
    conv = Conversation(server.client(), "Hello", preload=[], ctx_size=2000, max_tokens=500)
    conv.add_result("y" * 6000)
    with pytest.raises(ContextFullError):
        conv.next_step()
    assert server.requests == []


def test_conversation_limits_repeated_loading(server: ScriptedServer) -> None:
    names = ["eeg.epoch", "eeg.erp", "eeg.tfr"]
    server.replies += [{"action": "load_tools", "tools": [n]} for n in names]
    with pytest.raises(StepError, match="loaded tools 3 times"):
        Conversation(server.client(), "Hello", preload=[], max_loads=2).next_step()


class DoubleIn(BaseModel):
    n: int


class DoubleOut(BaseModel):
    doubled: int


def double(params: DoubleIn) -> DoubleOut:
    """Double a number."""
    return DoubleOut(doubled=2 * params.n)


def test_conversation_with_a_registry_of_its_own(server: ScriptedServer) -> None:
    tools = ToolRegistry()
    tools.register(make_tool(double, name="math.double", risk=Risk.READ))
    server.replies += [
        {"action": "load_tools", "tools": ["math.double"]},
        _call("math.double", n=4),
    ]
    conv = Conversation(server.client(), "Double 4", tools=tools)
    assert conv.loaded == []  # no keywords for this tool, and no dataset tool to fall back on
    assert conv.next_step() == CallTool(tool="math.double", arguments={"n": 4})
    assert "- math.double: Double a number." in server.requests[0]["messages"][0]["content"]


# --- the real model ----------------------------------------------------------------------


@pytest.mark.llama
def test_real_model(tmp_path: Path) -> None:
    from braino.runtime import start_server

    with start_server() as llama:
        client = ChatClient.for_server(llama)
        conv = Conversation(client, "Check the data quality of subject 003 in the oddball task.")
        step = conv.next_step()
        if isinstance(step, CallTool) and step.tool == "dataset.inspect":  # a fair first step
            conv.add_result({"subjects": ["001", "002", "003"], "tasks": ["oddball"]})
            step = conv.next_step()
        assert isinstance(step, CallTool) and step.tool == "eeg.qc_report"
        assert step.arguments["subject"] == "003"
        conv.add_result({"error": "MissingStepError: run eeg.preprocess first"})
        assert "eeg.preprocess" in [t.name for t in conv.loaded]
        step = conv.next_step()
        assert isinstance(step, CallTool) and step.tool == "eeg.preprocess"
        usage = conv.replies[-1].usage
        assert usage.cached_tokens and usage.cached_tokens > usage.prompt_tokens / 2
