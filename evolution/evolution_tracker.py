"""
evolution_tracker.py — Query and summarise evolution history.

Provides higher-level analytics on top of the raw JSONL written by SePOEngine.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class EvolutionTracker:
    """
    Read-only view over the SePO evolution history file.

    Parameters
    ----------
    history_path:
        Path to the JSONL produced by :class:`SePOEngine`.
    """

    def __init__(self, history_path: str = "logs/evolution_history.jsonl") -> None:
        self.history_path = history_path

    # ------------------------------------------------------------------ #
    # Raw access                                                          #
    # ------------------------------------------------------------------ #
    def all_records(self) -> List[Dict[str, Any]]:
        """Return every evolution record."""
        if not os.path.exists(self.history_path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return evolution records for a specific agent."""
        return [r for r in self.all_records() if r.get("agent_id") == agent_id]

    def latest(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent evolution record for *agent_id*."""
        records = self.for_agent(agent_id)
        return records[-1] if records else None

    # ------------------------------------------------------------------ #
    # Analytics                                                           #
    # ------------------------------------------------------------------ #
    def evolution_count_per_agent(self) -> Dict[str, int]:
        """Return how many times each agent has been evolved."""
        counter: Counter = Counter()
        for r in self.all_records():
            counter[r.get("agent_id", "unknown")] += 1
        return dict(counter)

    def anomaly_type_distribution(self) -> Dict[str, int]:
        """Return how often each anomaly type triggered an evolution."""
        counter: Counter = Counter()
        for r in self.all_records():
            counter[r.get("anomaly_type", "unknown")] += 1
        return dict(counter)

    def evolution_timeline(self) -> List[Dict[str, Any]]:
        """Return a time-sorted list of (timestamp, agent_id, anomaly_type)."""
        records = self.all_records()
        timeline = [
            {
                "timestamp": r.get("timestamp"),
                "agent_id": r.get("agent_id"),
                "anomaly_type": r.get("anomaly_type"),
                "method": r.get("method"),
            }
            for r in records
        ]
        return sorted(timeline, key=lambda x: x.get("timestamp") or "")

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary dict for the dashboard."""
        records = self.all_records()
        return {
            "total_evolutions": len(records),
            "agents_evolved": list({r.get("agent_id") for r in records}),
            "anomaly_type_distribution": self.anomaly_type_distribution(),
            "evolution_count_per_agent": self.evolution_count_per_agent(),
        }
