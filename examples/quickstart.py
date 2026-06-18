# -*- coding: utf-8 -*-
"""
examples/quickstart.py -- Minimal working demo (no LLM required).

Run from the project root::

    python examples/quickstart.py

This shows:
1. Registering multiple domain agents
2. Running queries through the full MASC + SePO pipeline
3. Injecting a bad agent to trigger MASC corrections
4. Viewing aggregate stats
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.agent_connector import AgentConnector
from examples.agents import FinanceAgent, HealthAgent, LegalAgent, SentimentAgent, SummaryAgent, EchoAgent
from connectors.base_agent import BaseAgent
from typing import Any


# ─── Optional: plug in a real LLM ─────────────────────────────────────
# Uncomment ONE of these to enable LLM-assisted corrections and evolution:
#
# from evolution.llm_protocol import OllamaAdapter
# llm = OllamaAdapter(model="llama3")          # local, free
#
# from evolution.llm_protocol import AnthropicAdapter
# llm = AnthropicAdapter(api_key="sk-ant-…")
#
# from evolution.llm_protocol import OpenAIAdapter
# llm = OpenAIAdapter(api_key="sk-…")
#
# from evolution.llm_protocol import GeminiAdapter
# llm = GeminiAdapter(api_key="AIza…")
#
llm = None  # heuristic SePO — works without any API keys


# ─── 1. Build connector ────────────────────────────────────────────────
connector = AgentConnector(
    llm_client=llm,
    anomaly_threshold=2,  # evolve after 2 consecutive anomalies
)

# ─── 2. Register agents ────────────────────────────────────────────────
connector.register_many([
    EchoAgent(),
    FinanceAgent(),
    HealthAgent(),
    LegalAgent(),
    SentimentAgent(),
    SummaryAgent(),
])


# ─── 3. A deliberately broken agent to demo MASC ──────────────────────
class BrokenFinanceAgent(BaseAgent):
    """Always returns a string instead of the expected JSON object."""

    @property
    def agent_id(self) -> str:
        return "broken_finance"

    @property
    def description(self) -> str:
        return "Intentionally broken — returns wrong type to trigger MASC."

    @property
    def output_schema(self):
        return FinanceAgent().output_schema

    def generate(self, query: str, **kwargs: Any) -> Any:
        return "I am not JSON"  # ← will trigger type_mismatch anomaly

connector.register(BrokenFinanceAgent())


# ─── 4. Run queries ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MASC / SePO Quickstart Demo")
print("=" * 60)

queries = [
    ("finance",        "What is the current GDP forecast?"),
    ("health",         "Patient age 45, complains of chest pain"),
    ("legal",          "Review NDA clause 4 for GDPR compliance"),
    ("sentiment",      "The new product launch was absolutely fantastic!"),
    ("summary",        "Summarize the Q3 earnings call transcript"),
    ("echo",           "hello framework!"),
    ("broken_finance", "What is inflation?"),
    ("broken_finance", "What is the interest rate?"),   # triggers SePO
]

for agent_id, query in queries:
    result = connector.run(agent_id, query)
    status = "[CORRECTED]" if result["corrected"] else ("[ERROR]" if result["error"] else "[OK]")
    print(f"\n{status} Agent: {agent_id}")
    print(f"  Query   : {query}")
    print(f"  Anomaly : {result['anomaly']}")
    print(f"  Latency : {result['latency_ms']} ms")
    output = result["output"]
    if isinstance(output, dict):
        import json
        print(f"  Output  : {json.dumps(output)[:300]}")
    else:
        print(f"  Output  : {str(output)[:200]}")


# ─── 5. Stats ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Aggregate Statistics")
print("=" * 60)
import json
stats = connector.stats()
print(json.dumps(stats, indent=2))
