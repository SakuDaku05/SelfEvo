"""
examples/agents.py — Ready-made example agents covering common domains.

These agents are fully functional WITHOUT any LLM — they use simple
rule-based or template logic so you can test the framework immediately.

To upgrade an agent to use a real LLM, subclass it and override
``generate()`` to call your preferred provider.

Included agents
---------------
* EchoAgent           — Returns the query as-is (testing/debugging)
* FinanceAgent        — Returns mock financial metrics
* HealthAgent         — Returns mock clinical summaries
* LegalAgent          — Returns mock contract analysis
* SentimentAgent      — Returns mock sentiment scores (0–1)
* SummaryAgent        — Returns a plain-text summary
* LLMBackedAgent      — Drop-in wrapper: any LLM + schema = full framework support
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.base_agent import BaseAgent


# ======================================================================= #
# Echo (for testing)                                                       #
# ======================================================================= #

class EchoAgent(BaseAgent):
    """Returns the query unchanged.  Useful for integration tests."""

    @property
    def agent_id(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the query back verbatim."

    def generate(self, query: str, **kwargs: Any) -> str:
        return query


# ======================================================================= #
# Finance                                                                  #
# ======================================================================= #

class FinanceAgent(BaseAgent):
    """
    Returns mock financial macro metrics.

    Output schema enforces:
    - gdp_growth_pct   : float in [-20, 20]
    - inflation_pct    : float in [0, 100]
    - interest_rate_pct: float in [0, 30]
    - sentiment        : enum [bearish, neutral, bullish]
    - summary          : non-empty string
    """

    @property
    def agent_id(self) -> str:
        return "finance"

    @property
    def description(self) -> str:
        return "Returns macro financial metrics and market sentiment."

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["gdp_growth_pct", "inflation_pct", "interest_rate_pct", "sentiment", "summary"],
            "properties": {
                "gdp_growth_pct":    {"type": "number", "minimum": -20, "maximum": 20},
                "inflation_pct":     {"type": "number", "minimum": 0, "maximum": 100},
                "interest_rate_pct": {"type": "number", "minimum": 0, "maximum": 30},
                "sentiment":         {"type": "string", "enum": ["bearish", "neutral", "bullish"]},
                "summary":           {"type": "string"},
            },
        }

    def generate(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "gdp_growth_pct":    round(random.uniform(-2, 4), 2),
            "inflation_pct":     round(random.uniform(1, 8), 2),
            "interest_rate_pct": round(random.uniform(0.5, 6), 2),
            "sentiment":         random.choice(["bearish", "neutral", "bullish"]),
            "summary":           f"Financial analysis for: {query}",
        }


# ======================================================================= #
# Health                                                                   #
# ======================================================================= #

class HealthAgent(BaseAgent):
    """
    Returns mock clinical / health metrics.

    Output schema enforces:
    - heart_rate_bpm   : integer in [40, 200]
    - bp_systolic      : integer in [80, 200]
    - bp_diastolic     : integer in [40, 130]
    - risk_level       : enum [low, moderate, high, critical]
    - recommendation   : non-empty string
    """

    @property
    def agent_id(self) -> str:
        return "health"

    @property
    def description(self) -> str:
        return "Returns clinical vitals and risk assessment."

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["heart_rate_bpm", "bp_systolic", "bp_diastolic", "risk_level", "recommendation"],
            "properties": {
                "heart_rate_bpm": {"type": "integer", "minimum": 40, "maximum": 200},
                "bp_systolic":    {"type": "integer", "minimum": 80, "maximum": 200},
                "bp_diastolic":   {"type": "integer", "minimum": 40, "maximum": 130},
                "risk_level":     {"type": "string", "enum": ["low", "moderate", "high", "critical"]},
                "recommendation": {"type": "string"},
            },
        }

    def generate(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "heart_rate_bpm": random.randint(60, 100),
            "bp_systolic":    random.randint(110, 140),
            "bp_diastolic":   random.randint(70, 90),
            "risk_level":     random.choice(["low", "moderate"]),
            "recommendation": f"Health assessment for: {query}",
        }


# ======================================================================= #
# Legal                                                                    #
# ======================================================================= #

class LegalAgent(BaseAgent):
    """
    Returns mock contract / legal analysis.

    Output schema enforces:
    - compliance_score  : float in [0, 1]
    - flagged_clauses   : array (minItems: 0)
    - jurisdiction      : non-empty string
    - verdict           : enum [approved, review_required, rejected]
    - explanation       : non-empty string
    """

    @property
    def agent_id(self) -> str:
        return "legal"

    @property
    def description(self) -> str:
        return "Returns contract compliance analysis and clause flags."

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["compliance_score", "flagged_clauses", "jurisdiction", "verdict", "explanation"],
            "properties": {
                "compliance_score": {"type": "number", "minimum": 0, "maximum": 1},
                "flagged_clauses":  {"type": "array", "items": {"type": "string"}, "minItems": 0},
                "jurisdiction":     {"type": "string"},
                "verdict":          {"type": "string", "enum": ["approved", "review_required", "rejected"]},
                "explanation":      {"type": "string"},
            },
        }

    def generate(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "compliance_score": round(random.uniform(0.6, 1.0), 2),
            "flagged_clauses":  random.choice([[], ["Clause 4.2: Limitation of liability"], ["Clause 7: Arbitration requirement"]]),
            "jurisdiction":     "United States (Federal)",
            "verdict":          random.choice(["approved", "review_required"]),
            "explanation":      f"Legal analysis for: {query}",
        }


# ======================================================================= #
# Sentiment                                                                #
# ======================================================================= #

class SentimentAgent(BaseAgent):
    """
    Returns a sentiment score + label for any text query.

    Output schema enforces:
    - score  : float in [0, 1]
    - label  : enum [negative, neutral, positive]
    - tokens : array minItems 1
    """

    @property
    def agent_id(self) -> str:
        return "sentiment"

    @property
    def description(self) -> str:
        return "Returns sentiment score and label for a given text."

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["score", "label", "tokens"],
            "properties": {
                "score":  {"type": "number", "minimum": 0, "maximum": 1},
                "label":  {"type": "string", "enum": ["negative", "neutral", "positive"]},
                "tokens": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
        }

    def generate(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        score = round(random.random(), 3)
        label = "positive" if score > 0.6 else ("negative" if score < 0.4 else "neutral")
        return {
            "score":  score,
            "label":  label,
            "tokens": query.split()[:10],
        }


# ======================================================================= #
# Summary (plain text, no schema)                                         #
# ======================================================================= #

class SummaryAgent(BaseAgent):
    """
    Returns a plain-text summary.  No schema — MASC checks it's non-empty.
    """

    @property
    def agent_id(self) -> str:
        return "summary"

    @property
    def description(self) -> str:
        return "Returns a plain-text summary of the query."

    # No output_schema → MASCValidator only checks for non-empty string

    def generate(self, query: str, **kwargs: Any) -> str:
        return f"Summary: {query[:200]}…" if len(query) > 200 else f"Summary: {query}"


# ======================================================================= #
# LLM-backed generic agent                                                #
# ======================================================================= #

class LLMBackedAgent(BaseAgent):
    """
    Generic wrapper that turns any LLM client into a framework-compatible
    agent.  Pass your LLM client + a system prompt and it just works.

    Parameters
    ----------
    agent_id:       Unique identifier shown in logs and API.
    description:    One-line description.
    llm_client:     Object with ``chat(messages) -> str``.
    system_prompt:  Initial system prompt.
    output_schema:  Optional JSON Schema dict for MASC validation.

    Example::

        from evolution.llm_protocol import OllamaAdapter
        llm = OllamaAdapter(model="llama3")

        agent = LLMBackedAgent(
            agent_id="my_agent",
            description="My custom LLM agent",
            llm_client=llm,
            system_prompt="You are a helpful assistant. Always reply in JSON.",
            output_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
        )
        connector.register(agent)
    """

    def __init__(
        self,
        agent_id: str,
        description: str,
        llm_client: Any,
        system_prompt: str = "You are a helpful assistant.",
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._agent_id = agent_id
        self._description = description
        self.llm_client = llm_client
        self._system_prompt = system_prompt
        self._output_schema = output_schema

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def description(self) -> str:
        return self._description

    @property
    def output_schema(self) -> Optional[Dict[str, Any]]:
        return self._output_schema

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    def generate(self, query: str, **kwargs: Any) -> Any:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        raw = self.llm_client.chat(messages)
        # Try to parse JSON if the schema expects an object/array
        if self._output_schema and self._output_schema.get("type") in ("object", "array"):
            import json, re
            text = re.sub(r"```(?:json)?\s*", "", raw).strip("` \n")
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
        return raw
