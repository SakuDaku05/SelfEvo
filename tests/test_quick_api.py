"""Quick offline test for the new add_fn / add_llm_agent API."""
import sys
sys.path.insert(0, ".")

from connectors.agent_connector import AgentConnector
from connectors.quick_agent import FunctionAgent


# ── Test 1: add_fn with lambda ───────────────────────────────────────────
c = AgentConnector()
c.add_fn("echo", lambda q, **kw: q, description="test agent")
r = c.run("echo", "hello MASC")
assert r["output"] == "hello MASC", f"Expected 'hello MASC', got {r['output']}"
assert r["corrected"] is False
print("[PASS] add_fn with lambda")


# ── Test 2: Chaining add_fn calls ────────────────────────────────────────
c2 = AgentConnector()
c2 \
    .add_fn("step1", lambda q, **kw: {"x": 1},
            schema={"type":"object","required":["x"],"properties":{"x":{"type":"number"}}}) \
    .add_fn("step2", lambda q, **kw: {"y": 2},
            schema={"type":"object","required":["y"],"properties":{"y":{"type":"number"}}})
ids = [a["agent_id"] for a in c2.list_agents()]
assert "step1" in ids and "step2" in ids
print("[PASS] chained .add_fn()")


# ── Test 3: add_fn with schema — MASC validates ──────────────────────────
c3 = AgentConnector()
c3.add_fn(
    "typed",
    lambda q, **kw: {"score": 0.95, "label": "positive"},
    schema={
        "type": "object",
        "required": ["score", "label"],
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        },
    },
)
r3 = c3.run("typed", "test")
assert r3["corrected"] is False
assert r3["output"]["score"] == 0.95
print("[PASS] add_fn with schema validation")


# ── Test 4: add_fn with BAD output — MASC corrects ──────────────────────
c4 = AgentConnector()
c4.add_fn(
    "bad_agent",
    lambda q, **kw: "I am a string not JSON",  # wrong type
    schema={"type": "object", "required": ["x"], "properties": {"x": {"type": "number"}}},
)
r4 = c4.run("bad_agent", "test")
assert r4["corrected"] is True, "MASC should have corrected type_mismatch"
assert r4["anomaly"]["type"] == "type_mismatch"
assert isinstance(r4["output"], dict), "CorrectionAgent should return a dict skeleton"
print("[PASS] add_fn — bad output auto-corrected by MASC")


# ── Test 5: add_llm_agent with a mock LLM ───────────────────────────────
class MockLLM:
    def chat(self, messages):
        import json
        return json.dumps({"result": "mocked", "confidence": 0.9})

c5 = AgentConnector()
c5.add_llm_agent(
    "mock_llm",
    llm_client=MockLLM(),
    system="You are a mock.",
    user_template="Answer this: {query}",
    schema={
        "type": "object",
        "required": ["result", "confidence"],
        "properties": {
            "result":     {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
)
r5 = c5.run("mock_llm", "hello")
assert r5["corrected"] is False
assert r5["output"]["result"] == "mocked"
assert r5["output"]["confidence"] == 0.9
print("[PASS] add_llm_agent with mock LLM")


# ── Test 6: add_llm_agent — user_template substitution ───────────────────
received_messages = []

class CaptureLLM:
    def chat(self, messages):
        received_messages.extend(messages)
        return "plain response"

c6 = AgentConnector()
c6.add_llm_agent(
    "template_test",
    llm_client=CaptureLLM(),
    system="System instruction here.",
    user_template="CONTEXT: {query} — please help.",
)
c6.run("template_test", "my query")
user_msg = next(m for m in received_messages if m["role"] == "user")
assert "my query" in user_msg["content"]
assert "CONTEXT:" in user_msg["content"]
sys_msg = next(m for m in received_messages if m["role"] == "system")
assert sys_msg["content"] == "System instruction here."
print("[PASS] add_llm_agent — user_template interpolation correct")


# ── Test 7: FunctionAgent auto-parses JSON strings ───────────────────────
import json as _json
c7 = AgentConnector()
c7.add_fn(
    "json_str_agent",
    lambda q, **kw: _json.dumps({"answer": 42}),  # returns a JSON string
    schema={"type": "object", "required": ["answer"], "properties": {"answer": {"type": "number"}}},
    parse_json=True,
)
r7 = c7.run("json_str_agent", "test")
assert r7["corrected"] is False
assert r7["output"]["answer"] == 42
print("[PASS] FunctionAgent auto-parses JSON string output")


# ── Test 8: parse_json=False skips auto-parsing ──────────────────────────
c8 = AgentConnector()
c8.add_fn(
    "raw_str_agent",
    lambda q, **kw: "raw text",
    parse_json=False,  # do NOT try to parse
)
r8 = c8.run("raw_str_agent", "test")
assert r8["output"] == "raw text"
print("[PASS] parse_json=False keeps output as string")


print("\n All 8 quick-connect API tests PASSED.")
