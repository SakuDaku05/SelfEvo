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

import asyncio
import time
import traceback
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from interceptor.masc_validator import MASCValidator
from interceptor.correction_agent import CorrectionAgent
from evolution.sepo_engine import SePOEngine
from logs.logger import Logger
from connectors.base_agent import BaseAgent
from connectors.quick_agent import FunctionAgent
from connectors.memory import MarkdownMemory


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
        self._memory: Optional[MarkdownMemory] = None

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
    # Memory                                                              #
    # ------------------------------------------------------------------ #
    def use_memory(
        self,
        memory: Optional[MarkdownMemory] = None,
        path: str = "memory.md",
        max_turns: int = 50,
        inject_turns: int = 5,
    ) -> "AgentConnector":
        """
        Enable Claude-style markdown memory for all agents.

        Every ``run()`` / ``arun()`` / ``stream()`` call will:
          1. Inject the agent's recent conversation turns as context
          2. Record the query + output as a new turn
          3. Append MASC anomaly notes and SePO evolution events

        Parameters
        ----------
        memory:
            Pass an existing ``MarkdownMemory`` instance to share memory
            across multiple connectors.  Leave ``None`` to create a new one.
        path:
            Path to the .md file (used when creating a new MarkdownMemory).
        max_turns:
            Max turns per agent stored in the file.
        inject_turns:
            Recent turns to inject as context on each call.

        Returns self for chaining::

            connector.use_memory("agents_memory.md").add_llm_agent(...)
        """
        self._memory = memory or MarkdownMemory(
            path=path, max_turns=max_turns, inject_turns=inject_turns
        )
        return self

    def _apply_memory_context(self, agent_id: str, query: str) -> str:
        """Prepend memory context to the query if memory is enabled."""
        if not self._memory:
            return query
        ctx = self._memory.get_context_string(agent_id)
        if ctx:
            return f"{ctx}\n---\nCurrent query: {query}"
        return query

    def _record_memory_turn(self, agent_id: str, query: str, result: Dict) -> None:
        """Write user turn + agent turn + any MASC/SePO notes to memory."""
        if not self._memory:
            return
        self._memory.add_turn(agent_id, "user", query)
        if result.get("output") is not None:
            import json as _json
            out = result["output"]
            self._memory.add_turn(
                agent_id, "agent",
                _json.dumps(out, default=str) if isinstance(out, (dict, list)) else str(out)
            )
        if result.get("corrected") and result.get("anomaly"):
            self._memory.add_masc_note(
                agent_id,
                anomaly_type=result["anomaly"].get("type", "unknown"),
                detail=result["anomaly"].get("detail", ""),
                corrected=True,
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
        enriched_query = self._apply_memory_context(agent_id, query)

        # -- 1. Call the agent ------------------------------------------ #
        t0 = time.perf_counter()
        raw_output: Any = None
        error: Optional[str] = None
        try:
            raw_output = agent.generate(enriched_query, **(extra_context or {}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw_output = None

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = self._validate_and_evolve(
            agent, agent_id, query, raw_output, error, latency_ms
        )
        self._record_memory_turn(agent_id, query, result)
        return result

    async def arun(
        self,
        agent_id: str,
        query: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Async version of ``run()``.  Works with both sync and async agents.

        Use inside FastAPI, asyncio pipelines, or parallel fan-out::

            result = await connector.arun("my_agent", "query")

            # Parallel execution across multiple agents:
            import asyncio
            results = await asyncio.gather(
                connector.arun("agent_a", query),
                connector.arun("agent_b", query),
            )
        """
        if agent_id not in self._agents:
            raise KeyError(
                f"Agent '{agent_id}' is not registered. "
                f"Available: {list(self._agents)}"
            )

        agent = self._agents[agent_id]
        enriched_query = self._apply_memory_context(agent_id, query)

        t0 = time.perf_counter()
        raw_output: Any = None
        error: Optional[str] = None
        try:
            raw_output = await agent.agenerate(enriched_query, **(extra_context or {}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw_output = None

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        result = self._validate_and_evolve(
            agent, agent_id, query, raw_output, error, latency_ms
        )
        self._record_memory_turn(agent_id, query, result)
        return result

    def stream(
        self,
        agent_id: str,
        query: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streaming run: yields chunks as they arrive, then a final
        validation/correction event.

        Yields dicts with key ``event``:
          - ``{"event": "chunk", "text": str}``  — each text chunk
          - ``{"event": "done",  **result}``      — full MASC result dict

        Usage::

            for msg in connector.stream("my_agent", "query"):
                if msg["event"] == "chunk":
                    print(msg["text"], end="", flush=True)
                elif msg["event"] == "done":
                    print()  # newline
                    if msg["corrected"]:
                        print("[MASC corrected:", msg["anomaly"], "]")
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent '{agent_id}' is not registered.")

        agent = self._agents[agent_id]
        enriched_query = self._apply_memory_context(agent_id, query)

        t0 = time.perf_counter()
        buffer: List[str] = []
        error: Optional[str] = None

        try:
            for chunk in agent.stream_generate(enriched_query):
                buffer.append(chunk)
                yield {"event": "chunk", "text": chunk}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        assembled = "".join(buffer)
        # Try JSON parse on assembled output before MASC
        from connectors.quick_agent import FunctionAgent
        parsed = FunctionAgent._try_parse(assembled) if assembled else assembled
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = self._validate_and_evolve(
            agent, agent_id, query, parsed if parsed != assembled else assembled,
            error, latency_ms
        )
        self._record_memory_turn(agent_id, query, result)
        yield {"event": "done", **result}

    async def astream(
        self,
        agent_id: str,
        query: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async streaming run.  Same events as ``stream()`` but as an
        async generator — use with ``async for``::

            async for msg in connector.astream("my_agent", "query"):
                if msg["event"] == "chunk":
                    print(msg["text"], end="", flush=True)
                elif msg["event"] == "done":
                    if msg["corrected"]:
                        print("\n[MASC corrected]", msg["anomaly"])
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent '{agent_id}' is not registered.")

        agent = self._agents[agent_id]
        enriched_query = self._apply_memory_context(agent_id, query)

        t0 = time.perf_counter()
        buffer: List[str] = []
        error: Optional[str] = None

        try:
            async for chunk in agent.astream_generate(enriched_query):
                buffer.append(chunk)
                yield {"event": "chunk", "text": chunk}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        assembled = "".join(buffer)
        parsed = FunctionAgent._try_parse(assembled) if assembled else assembled
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = self._validate_and_evolve(
            agent, agent_id, query, parsed if parsed != assembled else assembled,
            error, latency_ms
        )
        self._record_memory_turn(agent_id, query, result)
        yield {"event": "done", **result}

    # ------------------------------------------------------------------ #
    # Shared validation + evolution core (used by all run variants)       #
    # ------------------------------------------------------------------ #
    def _validate_and_evolve(
        self,
        agent: BaseAgent,
        agent_id: str,
        query: str,
        raw_output: Any,
        error: Optional[str],
        latency_ms: float,
    ) -> Dict[str, Any]:
        anomaly: Optional[Dict] = None
        corrected_output: Any = raw_output
        was_corrected = False

        if error is None:
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

                if self._anomaly_counts[agent_id] >= self.anomaly_threshold:
                    new_prompt = self.sepo.evolve(
                        agent_id=agent_id,
                        system_prompt=agent.system_prompt,
                        anomaly=anomaly,
                        correction=corrected_output,
                    )
                    if new_prompt:
                        agent.on_evolution(new_prompt)
                        self._anomaly_counts[agent_id] = 0
                        if self._memory:
                            self._memory.add_sepo_event(
                                agent_id,
                                anomaly_type=anomaly.get("type", "unknown"),
                                method="llm" if self.sepo.llm_client else "heuristic",
                            )
            else:
                self._anomaly_counts[agent_id] = 0

        self.logger.log(
            agent_id, query, corrected_output,
            corrected=was_corrected,
            anomaly=anomaly,
            error=error,
            latency_ms=latency_ms,
        )

        return {
            "output":     corrected_output,
            "corrected":  was_corrected,
            "anomaly":    anomaly,
            "agent_id":   agent_id,
            "latency_ms": latency_ms,
            "error":      error,
        }

    # ------------------------------------------------------------------ #
    # Aggregate stats (used by dashboard / metrics endpoint)             #
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return self.logger.aggregate_stats()
