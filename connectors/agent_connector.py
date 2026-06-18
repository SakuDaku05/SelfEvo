"""
agent_connector.py — The central orchestrator.

Wire-up order:
    connector = AgentConnector(llm_client=my_llm)
    connector.register("finance", FinanceAgent())
    result = connector.run("finance", "What is the GDP forecast?")

The connector handles:
    1. Routing queries to the right agent
    2. MASC validation of raw outputs
    3. CorrectionAgent fallback on anomalies
    4. SePO prompt-evolution when anomalies are repeated
    5. Structured logging of every run
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from interceptor.masc_validator import MASCValidator
from interceptor.correction_agent import CorrectionAgent
from evolution.sepo_engine import SePOEngine
from logs.logger import Logger
from connectors.base_agent import BaseAgent
from connectors.quick_agent import FunctionAgent


class AgentConnector:
    """
    Central hub that connects agents with MASC validation and SePO evolution.

    Parameters
    ----------
    llm_client:
        Any callable or object exposing ``chat(messages) -> str``.
        This is passed to SePOEngine so it can rewrite prompts using the
        *caller's* LLM of choice rather than a hardcoded provider.
        Pass ``None`` to disable prompt-evolution (MASC corrections still run).
    log_path:
        Path to the JSONL log file (default ``logs/agent_logs.json``).
    anomaly_threshold:
        Number of consecutive anomalies on the same agent before SePO
        triggers a full prompt-evolution cycle (default 3).
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        log_path: str = "logs/agent_logs.json",
        anomaly_threshold: int = 3,
    ) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._anomaly_counts: Dict[str, int] = {}

        self.validator = MASCValidator()
        self.correction = CorrectionAgent(llm_client=llm_client)
        self.sepo = SePOEngine(llm_client=llm_client)
        self.logger = Logger(log_path=log_path)
        self.anomaly_threshold = anomaly_threshold

    # ------------------------------------------------------------------ #
    # Registration                                                        #
    # ------------------------------------------------------------------ #
    def register(self, agent: BaseAgent, agent_id: Optional[str] = None) -> None:
        """Register an agent.  ``agent_id`` defaults to ``agent.agent_id``."""
        aid = agent_id or agent.agent_id
        self._agents[aid] = agent
        self._anomaly_counts.setdefault(aid, 0)

    def register_many(self, agents: List[BaseAgent]) -> None:
        """Register multiple agents at once."""
        for a in agents:
            self.register(a)

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return metadata for every registered agent."""
        return [a.metadata() for a in self._agents.values()]

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def add_fn(
        self,
        agent_id: str,
        fn: Callable[..., Any],
        *,
        system: str = "",
        schema: Optional[Dict[str, Any]] = None,
        description: str = "",
        parse_json: bool = True,
    ) -> "AgentConnector":
        """
        Register a plain function as an agent in one line.

        Parameters
        ----------
        agent_id:    Unique name for the agent.
        fn:          Any callable: ``fn(query: str) -> Any``
        system:      System prompt (also used by SePO when evolving).
        schema:      JSON Schema for the output — MASC uses this to
                     auto-validate and auto-correct.  Omit for plain text.
        description: Short description shown in dashboard/API.
        parse_json:  Auto-parse JSON strings before validation (default True).

        Returns self so calls can be chained::

            connector \\
                .add_fn("step1", fn1, schema=s1) \\
                .add_fn("step2", fn2, schema=s2)
        """
        self.register(
            FunctionAgent(
                agent_id=agent_id,
                fn=fn,
                system_prompt=system,
                output_schema=schema,
                description=description,
                parse_json=parse_json,
            )
        )
        return self

    def add_llm_agent(
        self,
        agent_id: str,
        *,
        llm_client: Any,
        system: str,
        user_template: str = "{query}",
        schema: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> "AgentConnector":
        """
        Register an LLM-backed agent in one line.

        The agent calls ``llm_client.chat([system_msg, user_msg])``
        automatically. You only need to supply the prompts and schema.

        Parameters
        ----------
        agent_id:      Unique name.
        llm_client:    Any object with ``chat(messages) -> str``.
        system:        System prompt.
        user_template: Template for the user message. Use ``{query}``
                       as the placeholder, e.g.
                       ``"Analyse the following data:\n{query}"``
        schema:        JSON Schema for MASC validation (optional).
        description:   Short description.

        Example::

            connector.add_llm_agent(
                "summariser",
                llm_client=llm,
                system="Summarise the following in 3 sentences.",
                user_template="Text to summarise:\n{query}",
            )
        """
        _system   = system          # capture for closure
        _template = user_template
        _llm      = llm_client

        def _llm_fn(query: str, **kw: Any) -> Any:
            return _llm.chat([
                {"role": "system", "content": _system},
                {"role": "user",   "content": _template.format(query=query)},
            ])

        return self.add_fn(
            agent_id,
            fn=_llm_fn,
            system=system,
            schema=schema,
            description=description,
            parse_json=True,
        )

    # ------------------------------------------------------------------ #
    # Core run loop                                                       #
    # ------------------------------------------------------------------ #
    def run(
        self,
        agent_id: str,
        query: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute *query* through the named agent with full MASC + SePO.

        Returns a dict with keys:
            - ``output``        — final (possibly corrected) output
            - ``corrected``     — True if MASC applied a correction
            - ``anomaly``       — detected anomaly type, or None
            - ``agent_id``      — echoed for convenience
            - ``latency_ms``    — round-trip latency in milliseconds
            - ``error``         — exception message if agent raised, else None
        """
        if agent_id not in self._agents:
            raise KeyError(
                f"Agent '{agent_id}' is not registered. "
                f"Available: {list(self._agents)}"
            )

        agent = self._agents[agent_id]

        # -- 1. Call the agent ------------------------------------------ #
        t0 = time.perf_counter()
        raw_output: Any = None
        error: Optional[str] = None
        try:
            raw_output = agent.generate(query, **(extra_context or {}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw_output = None

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # -- 2. MASC validation ----------------------------------------- #
        anomaly: Optional[str] = None
        corrected_output: Any = raw_output
        was_corrected = False

        if error is None:
            # Auto-discover validation rules from agent's output_schema
            anomaly = self.validator.check(raw_output, schema=agent.output_schema)

            if anomaly:
                self._anomaly_counts[agent_id] = (
                    self._anomaly_counts.get(agent_id, 0) + 1
                )
                corrected_output = self.correction.fix(
                    raw_output, anomaly, schema=agent.output_schema
                )
                was_corrected = True
                agent.on_correction(raw_output, corrected_output, anomaly)

                # -- 3. SePO evolution ---------------------------------- #
                if self._anomaly_counts[agent_id] >= self.anomaly_threshold:
                    new_prompt = self.sepo.evolve(
                        agent_id=agent_id,
                        system_prompt=agent.system_prompt,
                        anomaly=anomaly,
                        correction=corrected_output,
                    )
                    if new_prompt:
                        agent.on_evolution(new_prompt)
                        self._anomaly_counts[agent_id] = 0  # reset counter
            else:
                # Successful run — reset consecutive anomaly counter
                self._anomaly_counts[agent_id] = 0

        # -- 4. Log ---------------------------------------------------- #
        run_record = {
            "agent_id": agent_id,
            "query": query,
            "raw_output": raw_output,
            "output": corrected_output,
            "corrected": was_corrected,
            "anomaly": anomaly,
            "error": error,
            "latency_ms": latency_ms,
        }
        self.logger.log(**run_record)

        return {
            "output": corrected_output,
            "corrected": was_corrected,
            "anomaly": anomaly,
            "agent_id": agent_id,
            "latency_ms": latency_ms,
            "error": error,
        }

    # ------------------------------------------------------------------ #
    # Aggregate stats (used by dashboard / metrics endpoint)             #
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return self.logger.aggregate_stats()
