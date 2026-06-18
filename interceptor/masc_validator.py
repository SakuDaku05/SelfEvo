"""
masc_validator.py — Multi-Aspect Schema Check (MASC).

MASC validates agent outputs against a hierarchy of rules:

    1. Built-in structural rules  (type checks, null checks, …)
    2. JSON-Schema rules          (auto-generated from agent.output_schema)
    3. Custom rule plugins        (registered via MASCValidator.add_rule())

This design means MASC is **not** tied to a specific domain.  You never
need to hand-write "if not output.startswith('{')".  Instead the agent
declares its schema and MASC infers what to check.

Anomaly taxonomy
----------------
Every anomaly is a structured dict::

    {
        "type":    "schema_violation" | "null_output" | "type_mismatch" | …,
        "detail":  "<human-readable explanation>",
        "field":   "<JSON pointer to offending field, if applicable>",
        "value":   <offending value>,
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence


# ======================================================================= #
# Rule protocol                                                            #
# ======================================================================= #

class ValidationRule:
    """
    Base class for a single MASC validation rule.

    Subclass and implement :meth:`check`.  Return ``None`` on pass, or an
    anomaly dict on failure.
    """

    name: str = "unnamed_rule"
    description: str = ""

    def check(
        self, output: Any, schema: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


# ======================================================================= #
# Built-in rules                                                           #
# ======================================================================= #

class NullOutputRule(ValidationRule):
    """Catches None / empty string outputs."""
    name = "null_output"
    description = "Output must not be None or an empty string."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if output is None or output == "":
            return {
                "type": "null_output",
                "detail": "Agent returned None or an empty string.",
                "field": None,
                "value": output,
            }
        return None


class TypeMismatchRule(ValidationRule):
    """Checks top-level type against schema['type']."""
    name = "type_mismatch"
    description = "Output top-level type must match schema['type']."

    _TYPE_MAP: Dict[str, type] = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema:
            return None
        expected_type = schema.get("type")
        if not expected_type:
            return None
        py_type = self._TYPE_MAP.get(expected_type)
        if py_type and not isinstance(output, py_type):
            return {
                "type": "type_mismatch",
                "detail": (
                    f"Expected JSON type '{expected_type}' "
                    f"but got '{type(output).__name__}'."
                ),
                "field": "/",
                "value": type(output).__name__,
            }
        return None


class RequiredFieldsRule(ValidationRule):
    """Verifies all required fields are present for object outputs."""
    name = "required_fields"
    description = "All required schema fields must be present in object output."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema or not isinstance(output, dict):
            return None
        required: List[str] = schema.get("required", [])
        missing = [f for f in required if f not in output]
        if missing:
            return {
                "type": "required_fields",
                "detail": f"Missing required fields: {missing}",
                "field": ", ".join(f"/{f}" for f in missing),
                "value": missing,
            }
        return None


class PropertyTypeRule(ValidationRule):
    """Validates types of individual properties against schema['properties']."""
    name = "property_types"
    description = "Property values must match their declared types."

    _TYPE_MAP: Dict[str, type] = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema or not isinstance(output, dict):
            return None
        properties: Dict = schema.get("properties", {})
        for field, spec in properties.items():
            if field not in output:
                continue  # handled by RequiredFieldsRule
            expected = spec.get("type")
            if not expected:
                continue
            py_type = self._TYPE_MAP.get(expected)
            if py_type and not isinstance(output[field], py_type):
                return {
                    "type": "property_type_mismatch",
                    "detail": (
                        f"Field '{field}': expected '{expected}' "
                        f"but got '{type(output[field]).__name__}'."
                    ),
                    "field": f"/{field}",
                    "value": output[field],
                }
        return None


class NumericRangeRule(ValidationRule):
    """Validates minimum/maximum constraints on numeric fields."""
    name = "numeric_range"
    description = "Numeric fields must satisfy min/max constraints."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema or not isinstance(output, dict):
            return None
        for field, spec in schema.get("properties", {}).items():
            if field not in output:
                continue
            value = output[field]
            if not isinstance(value, (int, float)):
                continue
            lo = spec.get("minimum")
            hi = spec.get("maximum")
            if lo is not None and value < lo:
                return {
                    "type": "numeric_range",
                    "detail": f"Field '{field}' value {value} < minimum {lo}.",
                    "field": f"/{field}",
                    "value": value,
                }
            if hi is not None and value > hi:
                return {
                    "type": "numeric_range",
                    "detail": f"Field '{field}' value {value} > maximum {hi}.",
                    "field": f"/{field}",
                    "value": value,
                }
        return None


class StringPatternRule(ValidationRule):
    """Validates 'pattern' constraints on string fields."""
    name = "string_pattern"
    description = "String fields must match their declared regex pattern."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema or not isinstance(output, dict):
            return None
        for field, spec in schema.get("properties", {}).items():
            pattern = spec.get("pattern")
            if not pattern or field not in output:
                continue
            if not re.match(pattern, str(output[field])):
                return {
                    "type": "string_pattern",
                    "detail": (
                        f"Field '{field}' value {output[field]!r} "
                        f"does not match pattern '{pattern}'."
                    ),
                    "field": f"/{field}",
                    "value": output[field],
                }
        return None


class EnumRule(ValidationRule):
    """Validates 'enum' constraints."""
    name = "enum_violation"
    description = "Enum fields must only contain declared values."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema or not isinstance(output, dict):
            return None
        for field, spec in schema.get("properties", {}).items():
            allowed = spec.get("enum")
            if allowed is None or field not in output:
                continue
            if output[field] not in allowed:
                return {
                    "type": "enum_violation",
                    "detail": (
                        f"Field '{field}' value {output[field]!r} "
                        f"not in allowed values {allowed}."
                    ),
                    "field": f"/{field}",
                    "value": output[field],
                }
        return None


class NonEmptyArrayRule(ValidationRule):
    """Validates 'minItems' on array fields."""
    name = "empty_array"
    description = "Array fields must meet minItems constraint."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if not schema or not isinstance(output, dict):
            return None
        for field, spec in schema.get("properties", {}).items():
            min_items = spec.get("minItems")
            if min_items is None or field not in output:
                continue
            if isinstance(output[field], list) and len(output[field]) < min_items:
                return {
                    "type": "empty_array",
                    "detail": (
                        f"Field '{field}' has {len(output[field])} items "
                        f"but requires at least {min_items}."
                    ),
                    "field": f"/{field}",
                    "value": len(output[field]),
                }
        return None


# ======================================================================= #
# String-output rules (agents that don't return structured data)          #
# ======================================================================= #

class StringOutputNotEmptyRule(ValidationRule):
    """For non-schema agents: string output must not be blank."""
    name = "string_not_empty"
    description = "Plain-text agent output must not be blank."

    def check(self, output: Any, schema: Optional[Dict]) -> Optional[Dict]:
        if schema is not None:
            return None  # schema agents are handled by other rules
        if isinstance(output, str) and not output.strip():
            return {
                "type": "empty_string_output",
                "detail": "Agent returned an empty or whitespace-only string.",
                "field": None,
                "value": output,
            }
        return None


# ======================================================================= #
# MASCValidator                                                            #
# ======================================================================= #

# Default rule stack — order matters (short-circuit on first failure)
_DEFAULT_RULES: List[ValidationRule] = [
    NullOutputRule(),
    TypeMismatchRule(),
    RequiredFieldsRule(),
    PropertyTypeRule(),
    NumericRangeRule(),
    StringPatternRule(),
    EnumRule(),
    NonEmptyArrayRule(),
    StringOutputNotEmptyRule(),
]


class MASCValidator:
    """
    Multi-Aspect Schema Check validator.

    Usage::

        validator = MASCValidator()
        anomaly = validator.check(agent_output, schema=agent.output_schema)
        if anomaly:
            print(anomaly["type"], anomaly["detail"])

    Adding a custom rule::

        class MyRule(ValidationRule):
            name = "my_rule"
            def check(self, output, schema):
                if "bad_word" in str(output):
                    return {"type": "my_rule", "detail": "Bad word found",
                            "field": None, "value": output}

        validator.add_rule(MyRule())

    Parameters
    ----------
    rules:
        Override the full rule stack.  Defaults to :data:`_DEFAULT_RULES`.
    fail_fast:
        If True, return on the first anomaly found.  If False, collect all
        anomalies and return the first one (future: return list).
    """

    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        fail_fast: bool = True,
    ) -> None:
        self._rules: List[ValidationRule] = list(rules or _DEFAULT_RULES)
        self.fail_fast = fail_fast

    def add_rule(self, rule: ValidationRule, position: int = -1) -> None:
        """Append or insert a custom rule."""
        if position == -1:
            self._rules.append(rule)
        else:
            self._rules.insert(position, rule)

    def remove_rule(self, name: str) -> None:
        """Remove a rule by name."""
        self._rules = [r for r in self._rules if r.name != name]

    def list_rules(self) -> List[str]:
        return [r.name for r in self._rules]

    def check(
        self,
        output: Any,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run all rules against *output*.

        Returns the first anomaly dict found, or ``None`` if everything
        passes.
        """
        for rule in self._rules:
            try:
                result = rule.check(output, schema)
            except Exception as exc:
                result = {
                    "type": "rule_error",
                    "detail": f"Rule '{rule.name}' raised: {exc}",
                    "field": None,
                    "value": None,
                }
            if result is not None:
                result["rule"] = rule.name
                return result
        return None

    def check_all(
        self,
        output: Any,
        schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all rules and return every anomaly found (not just the first).
        """
        anomalies = []
        for rule in self._rules:
            try:
                result = rule.check(output, schema)
            except Exception as exc:
                result = {
                    "type": "rule_error",
                    "detail": f"Rule '{rule.name}' raised: {exc}",
                    "field": None,
                    "value": None,
                }
            if result is not None:
                result["rule"] = rule.name
                anomalies.append(result)
        return anomalies
