"""
correction_agent.py — Applies corrections to anomalous agent outputs.

Correction strategy is chosen automatically based on anomaly type and the
agent's declared output_schema.  For complex corrections it can optionally
call the user's LLM (the same one they passed into AgentConnector) to
generate a proper fix rather than returning a hard-coded fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class CorrectionAgent:
    """
    Applies corrections to outputs flagged by MASC.

    Parameters
    ----------
    llm_client:
        Optional LLM client (same interface as SePOEngine).  When set, the
        correction agent can ask the LLM to produce a valid output instead of
        relying only on built-in heuristics.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.llm_client = llm_client

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def fix(
        self,
        output: Any,
        anomaly: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Attempt to correct *output* given the detected *anomaly*.

        Strategy waterfall:
            1. If ``llm_client`` is available → ask the LLM to produce a
               valid response (best quality).
            2. Otherwise → apply deterministic heuristics based on anomaly
               type and schema.
            3. If all else fails → return a minimal skeleton derived from
               the schema (so downstream code gets a safe value).
        """
        anomaly_type = anomaly.get("type", "unknown")

        # -- LLM-assisted correction ------------------------------------ #
        if self.llm_client is not None:
            try:
                return self._llm_fix(output, anomaly, schema)
            except Exception:
                pass  # fall through to heuristics

        # -- Heuristic corrections -------------------------------------- #
        return self._heuristic_fix(output, anomaly_type, schema)

    # ------------------------------------------------------------------ #
    # LLM-assisted path                                                   #
    # ------------------------------------------------------------------ #
    def _llm_fix(
        self,
        output: Any,
        anomaly: Dict[str, Any],
        schema: Optional[Dict[str, Any]],
    ) -> Any:
        schema_hint = json.dumps(schema, indent=2) if schema else "No schema defined."
        prompt = (
            "You are a data correction assistant.\n"
            f"The following output was flagged as invalid:\n\n"
            f"Output:\n{json.dumps(output, default=str)}\n\n"
            f"Anomaly detected:\n{json.dumps(anomaly, indent=2)}\n\n"
            f"Expected schema:\n{schema_hint}\n\n"
            "Return ONLY the corrected JSON output with no explanation."
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self.llm_client.chat(messages)
        # Try to parse JSON from the response
        return self._try_parse_json(raw) or raw

    # ------------------------------------------------------------------ #
    # Heuristic path                                                      #
    # ------------------------------------------------------------------ #
    def _heuristic_fix(
        self,
        output: Any,
        anomaly_type: str,
        schema: Optional[Dict[str, Any]],
    ) -> Any:
        handlers = {
            "null_output": self._fix_null,
            "type_mismatch": self._fix_type_mismatch,
            "required_fields": self._fix_required_fields,
            "property_type_mismatch": self._fix_property_types,
            "numeric_range": self._fix_numeric_range,
            "string_pattern": lambda o, s: o,  # can't guess pattern fix
            "enum_violation": self._fix_enum,
            "empty_array": self._fix_empty_array,
            "empty_string_output": lambda o, s: "",
            "rule_error": lambda o, s: o,
        }
        handler = handlers.get(anomaly_type, lambda o, s: o)
        return handler(output, schema)

    def _fix_null(self, output: Any, schema: Optional[Dict]) -> Any:
        if not schema:
            return ""
        return self._skeleton(schema)

    def _fix_type_mismatch(self, output: Any, schema: Optional[Dict]) -> Any:
        if not schema:
            return output
        expected = schema.get("type")
        if expected == "object":
            if isinstance(output, str):
                parsed = self._try_parse_json(output)
                if parsed is not None:
                    return parsed
            return self._skeleton(schema)
        if expected == "array":
            return []
        if expected == "string":
            return str(output)
        if expected in ("number", "integer"):
            try:
                return int(output) if expected == "integer" else float(output)
            except (TypeError, ValueError):
                return 0
        if expected == "boolean":
            return bool(output)
        return output

    def _fix_required_fields(self, output: Any, schema: Optional[Dict]) -> Any:
        if not isinstance(output, dict) or not schema:
            return output
        fixed = dict(output)
        for field in schema.get("required", []):
            if field not in fixed:
                field_schema = schema.get("properties", {}).get(field, {})
                fixed[field] = self._default_for(field_schema)
        return fixed

    def _fix_property_types(self, output: Any, schema: Optional[Dict]) -> Any:
        if not isinstance(output, dict) or not schema:
            return output
        fixed = dict(output)
        for field, spec in schema.get("properties", {}).items():
            if field not in fixed:
                continue
            expected = spec.get("type")
            if not expected:
                continue
            val = fixed[field]
            if expected == "string" and not isinstance(val, str):
                fixed[field] = str(val)
            elif expected in ("number", "integer") and not isinstance(val, (int, float)):
                try:
                    fixed[field] = int(val) if expected == "integer" else float(val)
                except (TypeError, ValueError):
                    fixed[field] = 0
            elif expected == "boolean" and not isinstance(val, bool):
                fixed[field] = bool(val)
        return fixed

    def _fix_numeric_range(self, output: Any, schema: Optional[Dict]) -> Any:
        if not isinstance(output, dict) or not schema:
            return output
        fixed = dict(output)
        for field, spec in schema.get("properties", {}).items():
            if field not in fixed or not isinstance(fixed[field], (int, float)):
                continue
            lo = spec.get("minimum")
            hi = spec.get("maximum")
            if lo is not None:
                fixed[field] = max(fixed[field], lo)
            if hi is not None:
                fixed[field] = min(fixed[field], hi)
        return fixed

    def _fix_enum(self, output: Any, schema: Optional[Dict]) -> Any:
        if not isinstance(output, dict) or not schema:
            return output
        fixed = dict(output)
        for field, spec in schema.get("properties", {}).items():
            allowed = spec.get("enum")
            if allowed and field in fixed and fixed[field] not in allowed:
                fixed[field] = allowed[0]  # pick first valid value
        return fixed

    def _fix_empty_array(self, output: Any, schema: Optional[Dict]) -> Any:
        if not isinstance(output, dict) or not schema:
            return output
        fixed = dict(output)
        for field, spec in schema.get("properties", {}).items():
            min_items = spec.get("minItems", 0)
            if field in fixed and isinstance(fixed[field], list):
                while len(fixed[field]) < min_items:
                    fixed[field].append(None)
        return fixed

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _try_parse_json(text: Any) -> Optional[Any]:
        if not isinstance(text, str):
            return None
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _skeleton(schema: Dict[str, Any]) -> Any:
        """Build a minimal valid object from a schema."""
        t = schema.get("type", "object")
        if t == "object":
            out: Dict[str, Any] = {}
            for field, spec in schema.get("properties", {}).items():
                out[field] = CorrectionAgent._default_for(spec)
            return out
        if t == "array":
            return []
        if t == "string":
            return ""
        if t in ("number", "integer"):
            return 0
        if t == "boolean":
            return False
        return None

    @staticmethod
    def _default_for(spec: Dict[str, Any]) -> Any:
        t = spec.get("type", "string")
        if t == "string":
            enum = spec.get("enum")
            return enum[0] if enum else ""
        if t in ("number", "integer"):
            lo = spec.get("minimum", 0)
            return lo
        if t == "boolean":
            return False
        if t == "array":
            return []
        if t == "object":
            return {}
        return None
