"""
logger.py — Structured JSONL logging + aggregate stats.

Every agent run is persisted as a single JSON line.  The Logger also
maintains an in-memory cache of recent entries for fast dashboard reads.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class Logger:
    """
    Append-only structured logger for agent runs.

    Parameters
    ----------
    log_path:
        Path to the JSONL file.  Parent directories are created if needed.
    max_cache:
        Number of most-recent entries to keep in memory (for fast reads).
    """

    def __init__(
        self,
        log_path: str = "logs/agent_logs.json",
        max_cache: int = 1000,
    ) -> None:
        self.log_path = log_path
        self.max_cache = max_cache
        self._lock = threading.Lock()
        self._cache: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Write                                                               #
    # ------------------------------------------------------------------ #
    def log(
        self,
        agent_id: str,
        query: str,
        output: Any,
        raw_output: Any = None,
        corrected: bool = False,
        anomaly: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
        **extra: Any,
    ) -> None:
        """Persist one run record."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "query": query,
            "output": output,
            "raw_output": raw_output,
            "corrected": corrected,
            "anomaly": anomaly,
            "error": error,
            "latency_ms": latency_ms,
            **extra,
        }
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            self._cache.append(entry)
            if len(self._cache) > self.max_cache:
                self._cache = self._cache[-self.max_cache :]

    # ------------------------------------------------------------------ #
    # Read                                                                #
    # ------------------------------------------------------------------ #
    def recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the *n* most recent log entries (from cache then disk)."""
        with self._lock:
            cache_copy = list(self._cache)
        if len(cache_copy) >= n:
            return cache_copy[-n:]
        # Load from disk as fallback
        return self._load_from_disk()[-n:]

    def all_logs(self) -> List[Dict[str, Any]]:
        """Load every log entry from disk."""
        return self._load_from_disk()

    def _load_from_disk(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    # ------------------------------------------------------------------ #
    # Aggregate stats (used by dashboard & /stats endpoint)             #
    # ------------------------------------------------------------------ #
    def aggregate_stats(self) -> Dict[str, Any]:
        """
        Compute per-agent and global statistics.

        Returns a dict with keys:
            - ``global``   — total runs, error rate, avg latency, correction rate
            - ``agents``   — per-agent breakdown of the same stats
        """
        logs = self.all_logs()
        if not logs:
            return {"global": {}, "agents": {}}

        agent_buckets: Dict[str, List[Dict]] = defaultdict(list)
        for entry in logs:
            agent_buckets[entry.get("agent_id", "unknown")].append(entry)

        def _bucket_stats(entries: List[Dict]) -> Dict[str, Any]:
            total = len(entries)
            errors = sum(1 for e in entries if e.get("error"))
            corrections = sum(1 for e in entries if e.get("corrected"))
            latencies = [e["latency_ms"] for e in entries if isinstance(e.get("latency_ms"), (int, float))]
            anomaly_types: Dict[str, int] = defaultdict(int)
            for e in entries:
                a = e.get("anomaly")
                if a:
                    t = a.get("type", "unknown") if isinstance(a, dict) else str(a)
                    anomaly_types[t] += 1
            return {
                "total_runs": total,
                "error_count": errors,
                "error_rate": round(errors / total, 4) if total else 0,
                "correction_count": corrections,
                "correction_rate": round(corrections / total, 4) if total else 0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "anomaly_type_counts": dict(anomaly_types),
            }

        return {
            "global": _bucket_stats(logs),
            "agents": {aid: _bucket_stats(entries) for aid, entries in agent_buckets.items()},
        }
