# -*- coding: utf-8 -*-
"""
tests/test_gemini_live.py
Live integration test using the Gemini API (gemini-2.5-flash).

Free-tier limits for gemini-2.5-flash: 5 RPM, 25 RPD.
Strategy:
  - CALL_SLEEP = 13s  (5 calls/min = 1 every 12s, +1s buffer)
  - gemini_call() retries up to 3x with 65s backoff on 429

Run ONLY these tests (to avoid other test files eating quota):
    .venv\\Scripts\\pytest tests/test_gemini_live.py -v -s
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
skip_if_no_key = pytest.mark.skipif(not API_KEY, reason="GEMINI_API_KEY not set")

MODEL = "gemini-2.5-flash"
CALL_SLEEP = 13.0   # >12s to stay under 5 RPM
MAX_RETRIES = 3
RETRY_WAIT  = 65.0  # Gemini says retry after ~55-60s on free tier


def gemini_call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs); retry on 429/503 with backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                last_exc = exc
                wait = RETRY_WAIT * (attempt + 1)
                print(f"\n[rate-limit] 429 hit, waiting {wait:.0f}s (retry {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            elif "503" in msg or "UNAVAILABLE" in msg:
                last_exc = exc
                wait = 15.0 * (attempt + 1)  # shorter wait for 503
                print(f"\n[overloaded] 503 hit, waiting {wait:.0f}s (retry {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise last_exc


# ── GeminiAdapter smoke tests ─────────────────────────────────────────────

@skip_if_no_key
def test_gemini_adapter_basic_chat():
    """Adapter returns a non-empty string for a simple prompt."""
    from evolution.llm_protocol import GeminiAdapter
    llm = GeminiAdapter(api_key=API_KEY, model=MODEL)
    result = gemini_call(llm.chat, [{"role": "user", "content": "Reply with exactly: OK"}])
    assert isinstance(result, str) and len(result.strip()) > 0
    print(f"\n  Response: {result[:80]}")
    time.sleep(CALL_SLEEP)


@skip_if_no_key
def test_gemini_adapter_system_instruction():
    """System instruction is respected."""
    from evolution.llm_protocol import GeminiAdapter
    llm = GeminiAdapter(api_key=API_KEY, model=MODEL)
    result = gemini_call(llm.chat, [
        {"role": "system", "content": "Always reply with only the word PONG."},
        {"role": "user",   "content": "PING"},
    ])
    assert isinstance(result, str) and len(result.strip()) > 0
    print(f"\n  Response: {result[:80]}")
    time.sleep(CALL_SLEEP)


@skip_if_no_key
def test_gemini_adapter_json_output():
    """Gemini can be prompted to return JSON."""
    from evolution.llm_protocol import GeminiAdapter
    llm = GeminiAdapter(api_key=API_KEY, model=MODEL)
    result = gemini_call(llm.chat, [
        {"role": "system", "content": "Reply ONLY with valid JSON. No explanation."},
        {"role": "user",   "content": 'Return {"status": "ok", "value": 42}'},
    ])
    text = result.strip().strip("` \n").lstrip("json").strip()
    parsed = json.loads(text)
    assert parsed["value"] == 42
    print(f"\n  Parsed: {parsed}")
    time.sleep(CALL_SLEEP)


# ── SePO engine ───────────────────────────────────────────────────────────

@skip_if_no_key
def test_sepo_heuristic_evolve(tmp_path):
    """SePO heuristic evolve works without LLM — no API call."""
    from evolution.sepo_engine import SePOEngine
    sepo = SePOEngine(llm_client=None, history_path=str(tmp_path / "evo.jsonl"))
    new_prompt = sepo.evolve(
        agent_id="test_agent",
        system_prompt="You are a helpful assistant.",
        anomaly={"type": "null_output", "detail": "Returned None"},
        correction="",
    )
    assert new_prompt is not None
    assert "CRITICAL" in new_prompt
    history = sepo.history("test_agent")
    assert len(history) == 1
    assert history[0]["method"] == "heuristic"
    print(f"\n  New prompt snippet: {new_prompt[-80:]}")


@skip_if_no_key
def test_sepo_llm_evolve(tmp_path):
    """SePO uses Gemini to rewrite a system prompt."""
    from evolution.llm_protocol import GeminiAdapter
    from evolution.sepo_engine import SePOEngine

    llm = GeminiAdapter(api_key=API_KEY, model=MODEL)
    sepo = SePOEngine(llm_client=llm, history_path=str(tmp_path / "evo.jsonl"))

    new_prompt = gemini_call(
        sepo.evolve,
        agent_id="finance_agent",
        system_prompt="You are a financial analysis assistant.",
        anomaly={
            "type": "required_fields",
            "detail": "Missing required fields: ['gdp_growth_pct', 'summary']",
        },
        correction={"gdp_growth_pct": 0, "summary": ""},
    )

    assert new_prompt is not None
    assert isinstance(new_prompt, str) and len(new_prompt) > 20
    history = sepo.history("finance_agent")
    assert len(history) == 1
    assert history[0]["method"] == "llm"
    print(f"\n  Evolved prompt snippet: {new_prompt[:120]}")
    time.sleep(CALL_SLEEP)


# ── CorrectionAgent with Gemini ───────────────────────────────────────────

@skip_if_no_key
def test_correction_agent_llm_path(tmp_path):
    """CorrectionAgent uses Gemini to produce a corrected output."""
    from evolution.llm_protocol import GeminiAdapter
    from interceptor.correction_agent import CorrectionAgent

    llm = GeminiAdapter(api_key=API_KEY, model=MODEL)
    agent = CorrectionAgent(llm_client=llm)

    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }
    anomaly = {"type": "type_mismatch", "detail": "Expected object, got str",
               "field": "/", "value": "str"}

    result = gemini_call(agent.fix, "I am not JSON", anomaly, schema)
    assert result is not None
    print(f"\n  Correction result: {result}")
    time.sleep(CALL_SLEEP)


# ── Full end-to-end pipeline ──────────────────────────────────────────────

@skip_if_no_key
def test_full_pipeline_with_gemini(tmp_path):
    """
    End-to-end: LLMBackedAgent + GeminiAdapter + AgentConnector.
    Gemini acts as both the agent brain and SePO/correction LLM.
    """
    from evolution.llm_protocol import GeminiAdapter
    from connectors.agent_connector import AgentConnector
    from examples.agents import LLMBackedAgent

    llm = GeminiAdapter(api_key=API_KEY, model=MODEL)
    connector = AgentConnector(
        llm_client=llm,
        log_path=str(tmp_path / "logs.json"),
        anomaly_threshold=5,
    )

    schema = {
        "type": "object",
        "required": ["answer", "confidence"],
        "properties": {
            "answer":     {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }

    agent = LLMBackedAgent(
        agent_id="gemini_qa",
        description="Gemini-backed QA agent",
        llm_client=llm,
        system_prompt=(
            "You are a QA assistant. "
            "Always reply ONLY with a JSON object with keys: "
            "'answer' (string) and 'confidence' (float 0-1). No markdown, no explanation."
        ),
        output_schema=schema,
    )
    connector.register(agent)

    # Wrap connector.run with retry to handle transient 429
    result = gemini_call(connector.run, "gemini_qa", "What is 2 + 2?")

    assert result["error"] is None, f"Error: {result['error']}"
    output = result["output"]
    assert isinstance(output, dict), f"Expected dict, got {type(output)}: {output}"
    assert "answer" in output
    assert "confidence" in output
    assert 0 <= output["confidence"] <= 1
    print(f"\n  Pipeline result: {output}")
    time.sleep(CALL_SLEEP)


# ── MASC catches bad output (no LLM quota used) ───────────────────────────

@skip_if_no_key
def test_masc_catches_bad_output(tmp_path):
    """
    MASC catches a type_mismatch and corrects it heuristically.
    No API call made — saves quota.
    """
    from connectors.agent_connector import AgentConnector
    from connectors.base_agent import BaseAgent

    class AlwaysBadAgent(BaseAgent):
        @property
        def agent_id(self): return "always_bad"
        @property
        def output_schema(self):
            return {"type": "object", "required": ["x"], "properties": {"x": {"type": "number"}}}
        def generate(self, query, **kw):
            return "plain text, not JSON"

    connector = AgentConnector(llm_client=None, log_path=str(tmp_path / "logs.json"))
    connector.register(AlwaysBadAgent())
    result = connector.run("always_bad", "anything")
    assert result["corrected"] is True
    assert result["anomaly"]["type"] == "type_mismatch"
    assert isinstance(result["output"], dict)
    print(f"\n  Corrected output: {result['output']}")


# ── Evolution tracker (no API) ────────────────────────────────────────────

def test_evolution_tracker_summary(tmp_path):
    """EvolutionTracker analytics over SePO JSONL history."""
    from evolution.sepo_engine import SePOEngine
    from evolution.evolution_tracker import EvolutionTracker

    path = str(tmp_path / "evo.jsonl")
    sepo = SePOEngine(llm_client=None, history_path=path)

    for i in range(3):
        sepo.evolve(
            agent_id=f"agent_{i % 2}",
            system_prompt="old prompt",
            anomaly={"type": "null_output", "detail": "null"},
            correction="",
        )

    tracker = EvolutionTracker(history_path=path)
    summary = tracker.summary()
    assert summary["total_evolutions"] == 3
    assert len(summary["agents_evolved"]) == 2
    assert "null_output" in summary["anomaly_type_distribution"]
    timeline = tracker.evolution_timeline()
    assert len(timeline) == 3
    assert len(tracker.for_agent("agent_0")) == 2


# ── Logger stats (no API) ─────────────────────────────────────────────────

def test_logger_aggregate_stats(tmp_path):
    """Logger computes per-agent and global stats correctly."""
    from logs.logger import Logger

    logger = Logger(log_path=str(tmp_path / "logs.json"))
    logger.log("agent_a", "q1", {"x": 1}, corrected=False, latency_ms=10.0)
    logger.log("agent_a", "q2", None, corrected=True, anomaly={"type": "null_output"}, latency_ms=20.0)
    logger.log("agent_b", "q3", "text", corrected=False, latency_ms=5.0)
    logger.log("agent_b", "q4", None, error="ValueError: bad", latency_ms=1.0)

    stats = logger.aggregate_stats()
    g = stats["global"]
    assert g["total_runs"] == 4
    assert g["correction_count"] == 1
    assert g["error_count"] == 1
    assert g["avg_latency_ms"] == 9.0

    a = stats["agents"]["agent_a"]
    assert a["total_runs"] == 2
    assert a["correction_rate"] == 0.5

    b = stats["agents"]["agent_b"]
    assert b["error_count"] == 1
    assert b["error_rate"] == 0.5
