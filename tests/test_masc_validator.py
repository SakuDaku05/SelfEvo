# -*- coding: utf-8 -*-
"""
tests/test_masc_validator.py
Unit tests for the MASC validation rule stack.
No LLM / network calls — fully offline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from interceptor.masc_validator import (
    MASCValidator,
    ValidationRule,
    NullOutputRule,
    TypeMismatchRule,
    RequiredFieldsRule,
    PropertyTypeRule,
    NumericRangeRule,
    StringPatternRule,
    EnumRule,
    NonEmptyArrayRule,
    StringOutputNotEmptyRule,
)

SCHEMA = {
    "type": "object",
    "required": ["name", "score", "label", "tags"],
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["low", "medium", "high"]},
        "tags":  {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "code":  {"type": "string", "pattern": r"^[A-Z]{3}-\d{4}$"},
    },
}


# ── NullOutputRule ─────────────────────────────────────────────────────────

class TestNullOutputRule:
    def setup_method(self):
        self.rule = NullOutputRule()

    def test_none_flagged(self):
        result = self.rule.check(None, None)
        assert result is not None
        assert result["type"] == "null_output"

    def test_empty_string_flagged(self):
        result = self.rule.check("", None)
        assert result is not None
        assert result["type"] == "null_output"

    def test_valid_string_passes(self):
        assert self.rule.check("hello", None) is None

    def test_zero_passes(self):
        # 0 is a valid value, not None
        assert self.rule.check(0, None) is None

    def test_false_passes(self):
        assert self.rule.check(False, None) is None

    def test_empty_dict_passes(self):
        assert self.rule.check({}, None) is None


# ── TypeMismatchRule ───────────────────────────────────────────────────────

class TestTypeMismatchRule:
    def setup_method(self):
        self.rule = TypeMismatchRule()

    def test_string_when_object_expected(self):
        r = self.rule.check("not a dict", SCHEMA)
        assert r is not None
        assert r["type"] == "type_mismatch"

    def test_correct_type_passes(self):
        assert self.rule.check({"name": "x", "score": 0.5, "label": "low", "tags": ["a"]}, SCHEMA) is None

    def test_no_schema_passes(self):
        assert self.rule.check("anything", None) is None

    def test_array_when_object_expected(self):
        r = self.rule.check([1, 2, 3], SCHEMA)
        assert r is not None
        assert r["type"] == "type_mismatch"

    def test_number_schema(self):
        schema = {"type": "number"}
        assert self.rule.check(3.14, schema) is None
        r = self.rule.check("3.14", schema)
        assert r is not None


# ── RequiredFieldsRule ────────────────────────────────────────────────────

class TestRequiredFieldsRule:
    def setup_method(self):
        self.rule = RequiredFieldsRule()

    def test_missing_field_flagged(self):
        output = {"name": "x", "score": 0.5, "label": "low"}  # missing 'tags'
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "required_fields"
        assert "tags" in str(r["value"])

    def test_all_present_passes(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": ["a"]}
        assert self.rule.check(output, SCHEMA) is None

    def test_no_schema_passes(self):
        assert self.rule.check({}, None) is None

    def test_multiple_missing_fields(self):
        r = self.rule.check({}, SCHEMA)
        assert r is not None
        assert len(r["value"]) == 4  # all 4 required missing


# ── PropertyTypeRule ──────────────────────────────────────────────────────

class TestPropertyTypeRule:
    def setup_method(self):
        self.rule = PropertyTypeRule()

    def test_wrong_property_type(self):
        output = {"name": 123, "score": 0.5, "label": "low", "tags": ["a"]}
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "property_type_mismatch"
        assert r["field"] == "/name"

    def test_correct_types_passes(self):
        output = {"name": "ok", "score": 0.5, "label": "low", "tags": ["a"]}
        assert self.rule.check(output, SCHEMA) is None

    def test_integer_as_number_passes(self):
        schema = {"type": "object", "properties": {"v": {"type": "number"}}}
        assert self.rule.check({"v": 42}, schema) is None


# ── NumericRangeRule ──────────────────────────────────────────────────────

class TestNumericRangeRule:
    def setup_method(self):
        self.rule = NumericRangeRule()

    def test_below_minimum(self):
        output = {"name": "x", "score": -0.1, "label": "low", "tags": ["a"]}
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "numeric_range"

    def test_above_maximum(self):
        output = {"name": "x", "score": 1.5, "label": "low", "tags": ["a"]}
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "numeric_range"

    def test_at_boundary_passes(self):
        output = {"name": "x", "score": 0.0, "label": "low", "tags": ["a"]}
        assert self.rule.check(output, SCHEMA) is None
        output["score"] = 1.0
        assert self.rule.check(output, SCHEMA) is None

    def test_within_range_passes(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": ["a"]}
        assert self.rule.check(output, SCHEMA) is None


# ── StringPatternRule ─────────────────────────────────────────────────────

class TestStringPatternRule:
    def setup_method(self):
        self.rule = StringPatternRule()

    def test_invalid_pattern(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": ["a"], "code": "abc-123"}
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "string_pattern"

    def test_valid_pattern_passes(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": ["a"], "code": "ABC-1234"}
        assert self.rule.check(output, SCHEMA) is None

    def test_missing_field_not_flagged(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": ["a"]}
        assert self.rule.check(output, SCHEMA) is None


# ── EnumRule ──────────────────────────────────────────────────────────────

class TestEnumRule:
    def setup_method(self):
        self.rule = EnumRule()

    def test_invalid_enum_value(self):
        output = {"name": "x", "score": 0.5, "label": "critical", "tags": ["a"]}
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "enum_violation"

    def test_valid_enum_passes(self):
        for label in ["low", "medium", "high"]:
            output = {"name": "x", "score": 0.5, "label": label, "tags": ["a"]}
            assert self.rule.check(output, SCHEMA) is None


# ── NonEmptyArrayRule ─────────────────────────────────────────────────────

class TestNonEmptyArrayRule:
    def setup_method(self):
        self.rule = NonEmptyArrayRule()

    def test_empty_array_when_min_items_1(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": []}
        r = self.rule.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "empty_array"

    def test_non_empty_array_passes(self):
        output = {"name": "x", "score": 0.5, "label": "low", "tags": ["a"]}
        assert self.rule.check(output, SCHEMA) is None


# ── StringOutputNotEmptyRule ──────────────────────────────────────────────

class TestStringOutputNotEmptyRule:
    def setup_method(self):
        self.rule = StringOutputNotEmptyRule()

    def test_blank_string_flagged(self):
        r = self.rule.check("   ", None)
        assert r is not None
        assert r["type"] == "empty_string_output"

    def test_non_empty_string_passes(self):
        assert self.rule.check("hello", None) is None

    def test_skipped_when_schema_present(self):
        # With a schema, this rule is not relevant (other rules handle it)
        assert self.rule.check("", SCHEMA) is None


# ── MASCValidator integration ─────────────────────────────────────────────

class TestMASCValidator:
    def setup_method(self):
        self.v = MASCValidator()

    def test_valid_output_passes(self):
        output = {"name": "Alice", "score": 0.8, "label": "high", "tags": ["x"]}
        assert self.v.check(output, SCHEMA) is None

    def test_null_fails_first(self):
        r = self.v.check(None, SCHEMA)
        assert r is not None
        assert r["type"] == "null_output"

    def test_type_mismatch_fails(self):
        r = self.v.check("not a dict", SCHEMA)
        assert r is not None
        assert r["type"] == "type_mismatch"

    def test_check_all_returns_multiple(self):
        # Missing required fields + wrong type
        output = {"name": 42}  # wrong type for name AND missing required fields
        anomalies = self.v.check_all(output, SCHEMA)
        assert len(anomalies) >= 2

    def test_custom_rule_added(self):
        class BanRule(ValidationRule):
            name = "ban_word"
            def check(self, output, schema):
                if isinstance(output, dict) and output.get("name") == "BANNED":
                    return {"type": "ban_word", "detail": "banned name", "field": "/name", "value": "BANNED"}

        v = MASCValidator()
        v.add_rule(BanRule())
        output = {"name": "BANNED", "score": 0.5, "label": "low", "tags": ["a"]}
        r = v.check(output, SCHEMA)
        assert r is not None
        assert r["type"] == "ban_word"

    def test_list_rules(self):
        rules = self.v.list_rules()
        assert "null_output" in rules
        assert "required_fields" in rules

    def test_remove_rule(self):
        v = MASCValidator()
        v.remove_rule("null_output")
        # Now None should pass the null rule
        r = v.check(None, None)
        # Some other rule may catch it, but not null_output specifically
        if r:
            assert r["type"] != "null_output"

    def test_no_schema_string_agent(self):
        assert self.v.check("A good response.", None) is None

    def test_no_schema_empty_string(self):
        r = self.v.check("", None)
        assert r is not None
        assert r["type"] == "null_output"  # catches empty string before string_not_empty
