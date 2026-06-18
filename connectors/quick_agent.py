"""
connectors/quick_agent.py — Zero-boilerplate agent wrapper.

Wraps any plain Python callable into a MASC/SePO compatible agent.
No subclassing required.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

from connectors.base_agent import BaseAgent


class FunctionAgent(BaseAgent):
    """
    Wraps a plain Python function (or lambda) as a framework agent.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent.
    fn:
        The callable that does the actual work.
        Signature: ``fn(query: str, **kwargs) -> Any``
    system_prompt:
        Optional system instruction forwarded to the LLM inside ``fn``
        (if you use the provided ``llm`` argument). Also used by SePO
        when evolving the prompt.
    output_schema:
        Optional JSON Schema dict. MASC uses it to auto-validate and
        auto-correct the output.
    description:
        One-line description shown in the dashboard and API.
    parse_json:
        If True (default), the agent tries to parse ``fn``'s string
        return value as JSON before MASC sees it. This removes the need
        for JSON parsing boilerplate inside every function.
    """

    def __init__(
        self,
        agent_id: str,
        fn: Callable[..., Any],
        system_prompt: str = "",
        output_schema: Optional[Dict[str, Any]] = None,
        description: str = "",
        parse_json: bool = True,
    ) -> None:
        self._agent_id      = agent_id
        self._fn            = fn
        self._system_prompt = system_prompt
        self._output_schema = output_schema
        self._description   = description
        self._parse_json    = parse_json

    # ── Identity ────────────────────────────────────────────────────── #
    @property
    def agent_id(self) -> str:        return self._agent_id
    @property
    def description(self) -> str:     return self._description
    @property
    def output_schema(self):          return self._output_schema
    @property
    def system_prompt(self) -> str:   return self._system_prompt
    @system_prompt.setter
    def system_prompt(self, v: str):  self._system_prompt = v

    # ── Core ────────────────────────────────────────────────────────── #
    def generate(self, query: str, **kwargs: Any) -> Any:
        result = self._fn(query, **kwargs)

        if self._parse_json and isinstance(result, str):
            result = self._try_parse(result)

        return result

    # ── Helper ──────────────────────────────────────────────────────── #
    @staticmethod
    def _try_parse(text: str) -> Any:
        clean = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")
        try:
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            return text   # return as-is; MASC will flag if schema expects object
