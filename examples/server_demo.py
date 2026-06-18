"""
examples/server_demo.py — Start the API server pre-loaded with example agents.

Run::

    python examples/server_demo.py

Then test with::

    curl -X POST http://localhost:8000/query \\
         -H "Content-Type: application/json" \\
         -d '{"agent_id": "finance", "query": "GDP forecast Q4"}'

    curl http://localhost:8000/stats
    curl http://localhost:8000/agents
    curl http://localhost:8000/evolution
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

from connectors.agent_connector import AgentConnector
from api.server import app, setup
from examples.agents import (
    EchoAgent, FinanceAgent, HealthAgent, LegalAgent,
    SentimentAgent, SummaryAgent,
)

# ── Optional LLM (uncomment to enable LLM-assisted evolution) ──────────
# from evolution.llm_protocol import OllamaAdapter
# llm = OllamaAdapter(model="llama3")
llm = None

connector = AgentConnector(llm_client=llm, anomaly_threshold=3)
connector.register_many([
    EchoAgent(),
    FinanceAgent(),
    HealthAgent(),
    LegalAgent(),
    SentimentAgent(),
    SummaryAgent(),
])

setup(connector)

if __name__ == "__main__":
    print("Starting MASC/SePO API server at http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
