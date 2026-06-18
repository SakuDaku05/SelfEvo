"""
base_agent.py — Abstract interface for all agents plugged into the framework.

To add a new agent:
    1. Subclass BaseAgent
    2. Implement `generate(query: str) -> Any`
    3. Optionally override `output_schema` to guide MASC auto-validation
    4. Register with AgentConnector via connector.register("my_agent", MyAgent())
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseAgent(abc.ABC):
    """Minimal interface every domain agent must satisfy."""

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #
    @property
    def agent_id(self) -> str:
        """Human-readable identifier (defaults to class name)."""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """One-line description shown in the dashboard and API metadata."""
        return ""

    # ------------------------------------------------------------------ #
    # Schema hint (optional but recommended)                              #
    # ------------------------------------------------------------------ #
    @property
    def output_schema(self) -> Optional[Dict[str, Any]]:
        """
        Return a JSON-Schema-compatible dict describing the expected output.

        MASC uses this to auto-generate validation rules.  Return ``None``
        to skip schema-driven validation (raw text agents, etc.).

        Example::

            {
                "type": "object",
                "required": ["answer"],
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            }
        """
        return None

    @property
    def system_prompt(self) -> str:
        """
        The current system prompt sent to the underlying LLM.

        SePO will update this field as it evolves the agent.  Store it
        somewhere persistent (e.g. a DB) if you want cross-process
        persistence.
        """
        return ""

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:  # type: ignore[override]
        pass  # subclasses that want evolution must implement this

    # ------------------------------------------------------------------ #
    # Core                                                                 #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def generate(self, query: str, **kwargs: Any) -> Any:
        """
        Run the agent against *query* and return its raw output.

        The return value can be anything (str, dict, list …).  MASC will
        validate it against :attr:`output_schema` and any registered rules.
        """

    # ------------------------------------------------------------------ #
    # Optional lifecycle hooks                                            #
    # ------------------------------------------------------------------ #
    def on_correction(self, original: Any, corrected: Any, anomaly: str) -> None:
        """Called after MASC applies a correction.  Override to react."""

    def on_evolution(self, new_system_prompt: str) -> None:
        """Called after SePO rewrites the system prompt.  Override to persist."""
        self.system_prompt = new_system_prompt

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                    #
    # ------------------------------------------------------------------ #
    def metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "description": self.description,
            "has_schema": self.output_schema is not None,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id!r}>"
