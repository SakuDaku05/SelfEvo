"""
server.py — FastAPI application for the MASC/SePO connector framework.

Endpoints
---------
POST /query
    Run a query through a registered agent with full validation + evolution.

GET  /agents
    List all registered agents and their metadata.

GET  /stats
    Per-agent and global success/error/latency statistics.

GET  /stats/{agent_id}
    Stats for a single agent.

GET  /logs
    Recent run log entries (paginated with ?n=N).

GET  /evolution
    Full evolution history.

GET  /evolution/{agent_id}
    Evolution history for a single agent.

GET  /health
    Liveness probe.

GET  /rules
    List active MASC validation rules.

Usage
-----
Start the server::

    uvicorn api.server:app --reload

Then register your agents in the startup event or via the Python SDK.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so imports work when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.agent_connector import AgentConnector
from connectors.base_agent import BaseAgent
from evolution.evolution_tracker import EvolutionTracker

# ======================================================================= #
# Global connector — populated in lifespan or by calling setup()          #
# ======================================================================= #

_connector: Optional[AgentConnector] = None
_tracker: Optional[EvolutionTracker] = None


def get_connector() -> AgentConnector:
    if _connector is None:
        raise RuntimeError(
            "No AgentConnector registered.  "
            "Call api.server.setup(connector) before starting the server."
        )
    return _connector


def setup(
    connector: AgentConnector,
    evolution_history_path: str = "logs/evolution_history.jsonl",
) -> None:
    """
    Register a pre-configured AgentConnector with the server.

    Call this from your application startup code::

        from api.server import setup, app
        from connectors.agent_connector import AgentConnector
        from evolution.llm_protocol import OllamaAdapter

        llm = OllamaAdapter(model="llama3")
        connector = AgentConnector(llm_client=llm)
        connector.register(MyAgent())
        setup(connector)

        # then: uvicorn api.server:app
    """
    global _connector, _tracker
    _connector = connector
    _tracker = EvolutionTracker(history_path=evolution_history_path)


# ======================================================================= #
# Lifespan                                                                 #
# ======================================================================= #

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Hook for startup / shutdown.  Override by calling setup() before start."""
    global _connector, _tracker
    if _connector is None:
        # Default: create an empty connector with no LLM (heuristic-only SePO)
        _connector = AgentConnector()
        _tracker = EvolutionTracker()
    yield
    # Shutdown cleanup (if needed in future)


# ======================================================================= #
# App                                                                      #
# ======================================================================= #

app = FastAPI(
    title="MASC / SePO Agent Framework",
    description=(
        "LLM-agnostic connector framework with automatic output validation "
        "(MASC) and self-evolving prompt optimization (SePO)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================= #
# Request / response models                                               #
# ======================================================================= #

class QueryRequest(BaseModel):
    agent_id: str
    query: str
    extra_context: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    agent_id: str
    output: Any
    corrected: bool
    anomaly: Optional[Dict[str, Any]]
    latency_ms: float
    error: Optional[str]


# ======================================================================= #
# Endpoints                                                               #
# ======================================================================= #

@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe."""
    connector = get_connector()
    return {
        "status": "ok",
        "registered_agents": len(connector.list_agents()),
    }


@app.get("/agents", tags=["Meta"])
async def list_agents():
    """Return metadata for every registered agent."""
    return {"agents": get_connector().list_agents()}


@app.get("/rules", tags=["Meta"])
async def list_rules():
    """Return the active MASC validation rule names."""
    return {"rules": get_connector().validator.list_rules()}


@app.post("/query", response_model=QueryResponse, tags=["Core"])
async def run_query(req: QueryRequest):
    """
    Route a query to the specified agent with MASC validation + SePO evolution.
    """
    connector = get_connector()
    try:
        result = connector.run(
            agent_id=req.agent_id,
            query=req.query,
            extra_context=req.extra_context,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(**result)


@app.get("/stats", tags=["Observability"])
async def stats():
    """Per-agent and global run statistics."""
    return get_connector().stats()


@app.get("/stats/{agent_id}", tags=["Observability"])
async def agent_stats(agent_id: str):
    """Statistics for a single agent."""
    all_stats = get_connector().stats()
    agent_data = all_stats.get("agents", {}).get(agent_id)
    if agent_data is None:
        raise HTTPException(status_code=404, detail=f"No stats for agent '{agent_id}'")
    return {"agent_id": agent_id, **agent_data}


@app.get("/logs", tags=["Observability"])
async def recent_logs(n: int = Query(default=50, ge=1, le=1000)):
    """Return the *n* most recent run log entries."""
    return {"logs": get_connector().logger.recent(n)}


@app.get("/evolution", tags=["Observability"])
async def evolution_history():
    """Return the full SePO evolution history."""
    global _tracker
    if _tracker is None:
        return {"evolutions": []}
    return {
        "summary": _tracker.summary(),
        "timeline": _tracker.evolution_timeline(),
    }


@app.get("/evolution/{agent_id}", tags=["Observability"])
async def agent_evolution(agent_id: str):
    """Return SePO evolution records for a specific agent."""
    global _tracker
    if _tracker is None:
        return {"evolutions": []}
    records = _tracker.for_agent(agent_id)
    return {"agent_id": agent_id, "evolutions": records}
