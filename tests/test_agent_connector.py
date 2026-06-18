# -*- coding: utf-8 -*-
"""
tests/test_agent_connector.py
Integration tests for AgentConnector (no network).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from connectors.agent_connector import AgentConnector
from connectors.base_agent import BaseAgent
from typing import Any, Dict, Optional


# ── Test agents ───────────────────────────────────────────────────────────

class GoodAgent(BaseAgent):
    @property
    def agent_id(self): return "good"
    @property
    def description(self): return "Always valid"
    @property
    def output_schema(self):
        return {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "number", "minimum": 0, "maximum": 100}},
        }
    def generate(self, query, **kw) -> Dict:
        return {"value": 42}


class BadTypeAgent(BaseAgent):
    @property
    def agent_id(self): return "bad_type"
    @property
    def output_schema(self):
        return {"type": "object", "required": ["value"],
                "properties": {"value": {"type": "number"}}}
    def generate(self, query, **kw) -> Any:
        return "I am a string not a dict"  # type_mismatch


class RaisingAgent(BaseAgent):
    @property
    def agent_id(self): return "raiser"
    def generate(self, query, **kw):
        raise ValueError("Something went wrong inside the agent")


class PlainTextAgent(BaseAgent):
    @property
    def agent_id(self): return "plain"
    def generate(self, query, **kw) -> str:
        return f"Answer: {query}"


class EmptyTextAgent(BaseAgent):
    @property
    def agent_id(self): return "empty"
    def generate(self, query, **kw) -> str:
        return ""  # triggers null_output rule


# ── Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def connector(tmp_path):
    log_file = str(tmp_path / "test_logs.json")
    c = AgentConnector(llm_client=None, log_path=log_file, anomaly_threshold=3)
    c.register(GoodAgent())
    c.register(BadTypeAgent())
    c.register(RaisingAgent())
    c.register(PlainTextAgent())
    c.register(EmptyTextAgent())
    return c


# ── Tests ─────────────────────────────────────────────────────────────────

class TestAgentConnectorBasic:

    def test_good_agent_passes(self, connector):
        r = connector.run("good", "test query")
        assert r["error"] is None
        assert r["corrected"] is False
        assert r["anomaly"] is None
        assert r["output"] == {"value": 42}

    def test_unknown_agent_raises(self, connector):
        with pytest.raises(KeyError, match="not registered"):
            connector.run("nonexistent", "query")

    def test_raising_agent_captures_error(self, connector):
        r = connector.run("raiser", "will fail")
        assert r["error"] is not None
        assert "ValueError" in r["error"]
        assert r["corrected"] is False

    def test_bad_type_triggers_correction(self, connector):
        r = connector.run("bad_type", "query")
        assert r["corrected"] is True
        assert r["anomaly"] is not None
        assert r["anomaly"]["type"] == "type_mismatch"
        # Heuristic correction produces a dict (skeleton)
        assert isinstance(r["output"], dict)

    def test_plain_text_agent_passes(self, connector):
        r = connector.run("plain", "hello")
        assert r["error"] is None
        assert r["corrected"] is False
        assert "Answer:" in r["output"]

    def test_empty_text_agent_corrected(self, connector):
        r = connector.run("empty", "query")
        assert r["corrected"] is True

    def test_latency_recorded(self, connector):
        r = connector.run("good", "query")
        assert isinstance(r["latency_ms"], float)
        assert r["latency_ms"] >= 0

    def test_result_has_all_keys(self, connector):
        r = connector.run("good", "q")
        assert set(r.keys()) == {"output", "corrected", "anomaly", "agent_id", "latency_ms", "error"}


class TestAgentConnectorRegistration:

    def test_list_agents(self, connector):
        agents = connector.list_agents()
        ids = [a["agent_id"] for a in agents]
        assert "good" in ids
        assert "bad_type" in ids

    def test_register_many(self, tmp_path):
        c = AgentConnector(log_path=str(tmp_path / "l.json"))
        c.register_many([GoodAgent(), PlainTextAgent()])
        assert c.get_agent("good") is not None
        assert c.get_agent("plain") is not None

    def test_get_agent_none_for_unknown(self, connector):
        assert connector.get_agent("xyz") is None

    def test_register_with_custom_id(self, tmp_path):
        c = AgentConnector(log_path=str(tmp_path / "l.json"))
        c.register(GoodAgent(), agent_id="custom_good")
        assert c.get_agent("custom_good") is not None


class TestAgentConnectorSePOThreshold:

    def test_anomaly_counter_increments(self, connector):
        for _ in range(2):
            connector.run("bad_type", "query")
        count = connector._anomaly_counts.get("bad_type", 0)
        assert count == 2

    def test_counter_resets_on_good_run(self, tmp_path):
        c = AgentConnector(log_path=str(tmp_path / "l.json"), anomaly_threshold=3)
        c.register(GoodAgent())
        c._anomaly_counts["good"] = 2  # pretend 2 prior anomalies
        c.run("good", "q")             # good run
        assert c._anomaly_counts["good"] == 0

    def test_sepo_triggered_after_threshold(self, tmp_path):
        """After anomaly_threshold consecutive anomalies, SePO evolve() is called."""
        evolutions = []

        class TrackingSePO:
            def evolve(self, **kw):
                evolutions.append(kw)
                return None  # heuristic returns None to skip prompt update

        from interceptor.masc_validator import MASCValidator
        from interceptor.correction_agent import CorrectionAgent
        from logs.logger import Logger

        c = AgentConnector.__new__(AgentConnector)
        c.validator = MASCValidator()
        c.correction = CorrectionAgent()
        c.sepo = TrackingSePO()
        c.logger = Logger(log_path=str(tmp_path / "l.json"))
        c._agents = {}
        c._anomaly_counts = {}
        c.anomaly_threshold = 2

        c.register(BadTypeAgent())
        c.run("bad_type", "q1")  # anomaly #1
        assert len(evolutions) == 0
        c.run("bad_type", "q2")  # anomaly #2 → threshold hit
        assert len(evolutions) == 1


class TestAgentConnectorStats:

    def test_stats_after_runs(self, connector):
        connector.run("good", "q1")
        connector.run("bad_type", "q2")
        stats = connector.stats()
        assert "global" in stats
        assert "agents" in stats
        g = stats["global"]
        assert g["total_runs"] >= 2

    def test_correction_rate_reflects_corrections(self, connector):
        connector.run("bad_type", "q")
        stats = connector.stats()
        # bad_type should have correction_rate = 1.0
        bt = stats["agents"].get("bad_type", {})
        assert bt.get("correction_rate", 0) == 1.0
