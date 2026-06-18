# -*- coding: utf-8 -*-
"""
tests/test_correction_agent.py
Unit tests for CorrectionAgent — all heuristic paths (no LLM).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from interceptor.correction_agent import CorrectionAgent

SCHEMA = {
    "type": "object",
    "required": ["answer", "confidence", "status"],
    "properties": {
        "answer":     {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "status":     {"type": "string", "enum": ["ok", "partial", "fail"]},
        "tags":       {"type": "array", "minItems": 1},
    },
}


def anomaly(t):
    return {"type": t, "detail": f"test anomaly: {t}", "field": None, "value": None}


class TestCorrectionAgentHeuristic:
    """All tests use no llm_client — exercises heuristic paths only."""

    def setup_method(self):
        self.agent = CorrectionAgent(llm_client=None)

    # ── null_output ───────────────────────────────────────────────────── #
    def test_null_fix_with_schema(self):
        result = self.agent.fix(None, anomaly("null_output"), SCHEMA)
        assert isinstance(result, dict)
        assert "answer" in result

    def test_null_fix_without_schema(self):
        result = self.agent.fix(None, anomaly("null_output"), None)
        assert result == ""

    # ── type_mismatch ─────────────────────────────────────────────────── #
    def test_string_to_object_via_json(self):
        output = '{"answer": "42", "confidence": 0.9, "status": "ok"}'
        result = self.agent.fix(output, anomaly("type_mismatch"), SCHEMA)
        assert isinstance(result, dict)
        assert result["answer"] == "42"

    def test_unparseable_string_returns_skeleton(self):
        result = self.agent.fix("not json at all", anomaly("type_mismatch"), SCHEMA)
        assert isinstance(result, dict)

    def test_type_mismatch_number_coercion(self):
        schema = {"type": "number"}
        result = self.agent.fix("3.14", anomaly("type_mismatch"), schema)
        assert result == 3.14

    def test_type_mismatch_bool(self):
        schema = {"type": "boolean"}
        result = self.agent.fix(1, anomaly("type_mismatch"), schema)
        assert result is True

    def test_type_mismatch_array(self):
        schema = {"type": "array"}
        result = self.agent.fix("not a list", anomaly("type_mismatch"), schema)
        assert result == []

    def test_type_mismatch_string(self):
        schema = {"type": "string"}
        result = self.agent.fix(42, anomaly("type_mismatch"), schema)
        assert result == "42"

    # ── required_fields ───────────────────────────────────────────────── #
    def test_adds_missing_required_fields(self):
        output = {"answer": "hello"}
        result = self.agent.fix(output, anomaly("required_fields"), SCHEMA)
        assert "confidence" in result
        assert "status" in result

    def test_existing_fields_preserved(self):
        output = {"answer": "keep me"}
        result = self.agent.fix(output, anomaly("required_fields"), SCHEMA)
        assert result["answer"] == "keep me"

    # ── property_type_mismatch ────────────────────────────────────────── #
    def test_converts_string_number_to_float(self):
        output = {"answer": "x", "confidence": "0.7", "status": "ok"}
        result = self.agent.fix(output, anomaly("property_type_mismatch"), SCHEMA)
        assert isinstance(result["confidence"], float)

    def test_converts_int_to_string(self):
        output = {"answer": 42, "confidence": 0.5, "status": "ok"}
        result = self.agent.fix(output, anomaly("property_type_mismatch"), SCHEMA)
        assert isinstance(result["answer"], str)

    # ── numeric_range ─────────────────────────────────────────────────── #
    def test_clamps_below_minimum(self):
        output = {"answer": "x", "confidence": -5.0, "status": "ok"}
        result = self.agent.fix(output, anomaly("numeric_range"), SCHEMA)
        assert result["confidence"] >= 0

    def test_clamps_above_maximum(self):
        output = {"answer": "x", "confidence": 99.0, "status": "ok"}
        result = self.agent.fix(output, anomaly("numeric_range"), SCHEMA)
        assert result["confidence"] <= 1

    # ── enum_violation ────────────────────────────────────────────────── #
    def test_replaces_invalid_enum_with_first_allowed(self):
        output = {"answer": "x", "confidence": 0.5, "status": "UNKNOWN"}
        result = self.agent.fix(output, anomaly("enum_violation"), SCHEMA)
        assert result["status"] == "ok"  # first allowed value

    # ── empty_array ───────────────────────────────────────────────────── #
    def test_pads_empty_array_to_min_items(self):
        output = {"answer": "x", "confidence": 0.5, "status": "ok", "tags": []}
        result = self.agent.fix(output, anomaly("empty_array"), SCHEMA)
        assert len(result["tags"]) >= 1

    # ── unknown anomaly ───────────────────────────────────────────────── #
    def test_unknown_anomaly_returns_original(self):
        output = {"answer": "x", "confidence": 0.5, "status": "ok"}
        result = self.agent.fix(output, anomaly("some_unknown_type"), SCHEMA)
        assert result == output

    # ── skeleton builder ─────────────────────────────────────────────── #
    def test_skeleton_has_all_required_keys(self):
        skeleton = CorrectionAgent._skeleton(SCHEMA)
        assert "answer" in skeleton
        assert "confidence" in skeleton
        assert "status" in skeleton

    def test_skeleton_status_is_first_enum(self):
        skeleton = CorrectionAgent._skeleton(SCHEMA)
        assert skeleton["status"] == "ok"

    # ── JSON strip helpers ────────────────────────────────────────────── #
    def test_parse_json_strips_code_fences(self):
        text = "```json\n{\"a\": 1}\n```"
        result = CorrectionAgent._try_parse_json(text)
        assert result == {"a": 1}

    def test_parse_json_invalid_returns_none(self):
        assert CorrectionAgent._try_parse_json("not json") is None

    def test_parse_json_non_string_returns_none(self):
        assert CorrectionAgent._try_parse_json(42) is None
