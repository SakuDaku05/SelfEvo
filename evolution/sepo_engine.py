"""
sepo_engine.py — Self-Evolving Prompt Optimizer (SePO).

SePO monitors repeated anomalies and uses your LLM to rewrite agent system
prompts so the same mistakes stop recurring.

Key design choices
------------------
* **LLM-agnostic**: works with any object that has a ``chat(messages) -> str``
  method (see ``evolution/llm_protocol.py`` for ready-made adapters).
* **No LLM required**: if no ``llm_client`` is passed, SePO falls back to a
  rule-based heuristic that injects explicit instructions about the anomaly.
* **Persistent**: evolution history is written to a JSONL file so you can
  review what changed and roll back if needed.
* **Thread-safe**: a ``threading.Lock`` protects concurrent calls.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class SePOEngine:
    """
    Self-Evolving Prompt Optimizer.

    Parameters
    ----------
    llm_client:
        Object with ``chat(messages: list[dict]) -> str``.
        Pass ``None`` to use the built-in heuristic evolver.
    history_path:
        Path to the JSONL file where evolution records are persisted.
    max_prompt_tokens:
        Soft cap on generated prompt length (passed as instruction to LLM).
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        history_path: str = "logs/evolution_history.jsonl",
        max_prompt_tokens: int = 512,
    ) -> None:
        self.llm_client = llm_client
        self.history_path = history_path
        self.max_prompt_tokens = max_prompt_tokens
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Core                                                                 #
    # ------------------------------------------------------------------ #
    def evolve(
        self,
        agent_id: str,
        system_prompt: str,
        anomaly: Dict[str, Any],
        correction: Any,
    ) -> Optional[str]:
        """
        Generate a new system prompt that should prevent *anomaly* from
        recurring.

        Returns the new prompt string, or ``None`` if evolution failed.
        """
        anomaly_type = anomaly.get("type", "unknown")
        anomaly_detail = anomaly.get("detail", "")

        if self.llm_client is not None:
            try:
                new_prompt = self._llm_evolve(
                    agent_id, system_prompt, anomaly_type, anomaly_detail, correction
                )
            except Exception:
                # LLM unavailable — fall back to heuristic silently
                new_prompt = self._heuristic_evolve(
                    system_prompt, anomaly_type, anomaly_detail
                )
        else:
            new_prompt = self._heuristic_evolve(
                system_prompt, anomaly_type, anomaly_detail
            )

        if new_prompt:
            self._persist(agent_id, system_prompt, new_prompt, anomaly)

        return new_prompt

    # ------------------------------------------------------------------ #
    # LLM-assisted evolution                                              #
    # ------------------------------------------------------------------ #
    def _llm_evolve(
        self,
        agent_id: str,
        system_prompt: str,
        anomaly_type: str,
        anomaly_detail: str,
        correction: Any,
    ) -> Optional[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer specializing in making "
                    "AI agents more reliable and consistent."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Agent ID: {agent_id}\n\n"
                    f"Current system prompt:\n'''\n{system_prompt}\n'''\n\n"
                    f"Recurring anomaly type: {anomaly_type}\n"
                    f"Anomaly detail: {anomaly_detail}\n"
                    f"Applied correction: {json.dumps(correction, default=str)}\n\n"
                    "Task: Rewrite the system prompt to prevent this anomaly from "
                    "occurring in future responses.  Rules:\n"
                    f"- Keep the prompt under {self.max_prompt_tokens} tokens\n"
                    "- Preserve all existing valid instructions\n"
                    "- Add explicit constraints that address the anomaly\n"
                    "- Output ONLY the new system prompt, no explanations"
                ),
            },
        ]
        # Let exceptions propagate — caller can retry on 429/503
        result = self.llm_client.chat(messages).strip()
        if not result:
            return None
        return result

    # ------------------------------------------------------------------ #
    # Heuristic (no-LLM) evolution                                       #
    # ------------------------------------------------------------------ #
    _HEURISTIC_PATCHES: Dict[str, str] = {
        "null_output": (
            "\n\nCRITICAL: Always return a non-empty response.  "
            "Never return null or an empty string."
        ),
        "type_mismatch": (
            "\n\nCRITICAL: Your response MUST match the expected data type exactly "
            "as specified in the task description."
        ),
        "required_fields": (
            "\n\nCRITICAL: Always include ALL required fields in your response.  "
            "Missing fields cause downstream failures."
        ),
        "property_type_mismatch": (
            "\n\nCRITICAL: Each field in your JSON response must have the exact "
            "data type specified in the schema (string, number, boolean, etc.)."
        ),
        "numeric_range": (
            "\n\nCRITICAL: All numeric values must stay within their specified "
            "minimum and maximum bounds."
        ),
        "string_pattern": (
            "\n\nCRITICAL: String fields must match the required format/pattern "
            "exactly (e.g. dates as YYYY-MM-DD, IDs as UUIDs, etc.)."
        ),
        "enum_violation": (
            "\n\nCRITICAL: Enum fields must only use values from the allowed list.  "
            "Do not invent new values."
        ),
        "empty_array": (
            "\n\nCRITICAL: Array fields must contain at least the minimum number "
            "of items specified in the schema."
        ),
        "empty_string_output": (
            "\n\nCRITICAL: Always provide a substantive response.  "
            "Never return an empty or whitespace-only reply."
        ),
    }

    def _heuristic_evolve(
        self, system_prompt: str, anomaly_type: str, anomaly_detail: str
    ) -> str:
        patch = self._HEURISTIC_PATCHES.get(
            anomaly_type,
            f"\n\nCRITICAL: Avoid the following error: {anomaly_detail}",
        )
        # De-duplicate — don't append the same patch twice
        if patch.strip() in system_prompt:
            return system_prompt
        return system_prompt + patch

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #
    def _persist(
        self,
        agent_id: str,
        old_prompt: str,
        new_prompt: str,
        anomaly: Dict[str, Any],
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "anomaly_type": anomaly.get("type"),
            "anomaly_detail": anomaly.get("detail"),
            "old_prompt": old_prompt,
            "new_prompt": new_prompt,
            "method": "llm" if self.llm_client else "heuristic",
        }
        with self._lock:
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # History query                                                       #
    # ------------------------------------------------------------------ #
    def history(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load and return evolution history, optionally filtered by agent."""
        if not os.path.exists(self.history_path):
            return []
        records = []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if agent_id is None or rec.get("agent_id") == agent_id:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
        return records
