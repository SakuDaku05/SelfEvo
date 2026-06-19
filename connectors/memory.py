"""
connectors/memory.py — Claude-style Markdown memory for agents.

Each agent gets its own section in a single .md file.
Stores conversation turns, MASC anomaly notes, and SePO evolution events.
Memory is injected into agent prompts as context on every run.

File format (memory.md)
-----------------------

# SelfEvo Agent Memory

## Agent: trend_researcher
*Last updated: 2025-06-19 10:30:00*

### Conversation
| Turn | Role | Content |
|------|------|---------|
| 2025-06-19 10:28 | user | Research AI trends |
| 2025-06-19 10:28 | agent | {"trends": ["LLM agents", ...]} |

### MASC Notes
- `2025-06-19 10:28` — anomaly `null_output` detected, corrected automatically.

### SePO Events
- `2025-06-19 10:29` — prompt evolved due to repeated `null_output` anomalies.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional


class MarkdownMemory:
    """
    Claude-style markdown memory file shared across all agents.

    Parameters
    ----------
    path:
        Path to the .md memory file (created if it doesn't exist).
    max_turns:
        Maximum conversation turns to keep per agent (oldest dropped first).
    inject_turns:
        How many recent turns to inject into agent context on each run.

    Usage
    -----
    ::

        from connectors.memory import MarkdownMemory

        memory = MarkdownMemory("memory.md", max_turns=50, inject_turns=5)
        connector = AgentConnector(llm_client=llm)
        connector.use_memory(memory)

        # Now every connector.run() / arun() automatically:
        #   1. Injects recent turns as context into the agent prompt
        #   2. Records the query and output as a new turn
        #   3. Logs MASC anomalies and SePO events as notes
    """

    _HEADER = "# SelfEvo Agent Memory\n\n"

    def __init__(
        self,
        path: str = "memory.md",
        max_turns: int = 50,
        inject_turns: int = 5,
    ) -> None:
        self.path = path
        self.max_turns = max_turns
        self.inject_turns = inject_turns
        self._lock = threading.Lock()
        self._data: Dict[str, Dict] = {}   # agent_id -> {turns, masc_notes, sepo_events}
        self._load()

    # ── Public API ─────────────────────────────────────────────────────── #

    def add_turn(self, agent_id: str, role: str, content: str) -> None:
        """Record a user query or agent response."""
        with self._lock:
            turns = self._agent(agent_id)["turns"]
            turns.append({
                "ts":      self._ts(),
                "role":    role,
                "content": str(content)[:500],   # cap length for readability
            })
            # Trim to max_turns
            if len(turns) > self.max_turns:
                self._agent(agent_id)["turns"] = turns[-self.max_turns:]
            self._save()

    def add_masc_note(
        self,
        agent_id: str,
        anomaly_type: str,
        detail: str = "",
        corrected: bool = True,
    ) -> None:
        """Record a MASC anomaly event."""
        with self._lock:
            note = f"`{self._ts()}` — anomaly `{anomaly_type}`"
            if detail:
                note += f": {detail[:120]}"
            if corrected:
                note += " — **corrected automatically**."
            else:
                note += " — correction failed."
            self._agent(agent_id)["masc_notes"].append(note)
            self._save()

    def add_sepo_event(
        self,
        agent_id: str,
        anomaly_type: str,
        method: str = "heuristic",
    ) -> None:
        """Record a SePO prompt evolution event."""
        with self._lock:
            event = (
                f"`{self._ts()}` — prompt evolved (`{method}`) "
                f"due to repeated `{anomaly_type}` anomalies."
            )
            self._agent(agent_id)["sepo_events"].append(event)
            self._save()

    def get_context_string(self, agent_id: str) -> str:
        """
        Returns a markdown-formatted context block to inject into a system prompt.
        Includes the N most recent turns and any MASC/SePO notes.
        """
        with self._lock:
            data = self._data.get(agent_id)
        if not data or (not data["turns"] and not data["masc_notes"]):
            return ""

        lines = [f"## Memory context for agent `{agent_id}`\n"]

        if data["turns"]:
            recent = data["turns"][-self.inject_turns:]
            lines.append("### Recent conversation turns")
            for t in recent:
                role_label = "User" if t["role"] == "user" else "Agent"
                lines.append(f"- **{role_label}** ({t['ts']}): {t['content']}")
            lines.append("")

        if data["masc_notes"]:
            lines.append("### Recent MASC notes (last 3)")
            for note in data["masc_notes"][-3:]:
                lines.append(f"- {note}")
            lines.append("")

        if data["sepo_events"]:
            lines.append("### SePO evolution history")
            for ev in data["sepo_events"][-3:]:
                lines.append(f"- {ev}")
            lines.append("")

        return "\n".join(lines)

    def get_messages(self, agent_id: str) -> List[Dict[str, str]]:
        """
        Returns recent turns as an OpenAI-style message list for
        direct injection into LLM chat history.
        """
        with self._lock:
            turns = self._data.get(agent_id, {}).get("turns", [])
        recent = turns[-self.inject_turns:]
        return [
            {
                "role":    "user" if t["role"] == "user" else "assistant",
                "content": t["content"],
            }
            for t in recent
        ]

    def clear_agent(self, agent_id: str) -> None:
        """Wipe memory for a specific agent."""
        with self._lock:
            if agent_id in self._data:
                del self._data[agent_id]
            self._save()

    def clear_all(self) -> None:
        """Wipe the entire memory file."""
        with self._lock:
            self._data = {}
            self._save()

    def summary(self) -> Dict:
        """Return a summary of stored memory."""
        with self._lock:
            return {
                agent_id: {
                    "turns":       len(v["turns"]),
                    "masc_notes":  len(v["masc_notes"]),
                    "sepo_events": len(v["sepo_events"]),
                }
                for agent_id, v in self._data.items()
            }

    # ── Internal ───────────────────────────────────────────────────────── #

    def _agent(self, agent_id: str) -> Dict:
        """Get or create the memory bucket for an agent."""
        if agent_id not in self._data:
            self._data[agent_id] = {
                "turns":       [],
                "masc_notes":  [],
                "sepo_events": [],
            }
        return self._data[agent_id]

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _save(self) -> None:
        """Serialize _data to markdown."""
        lines = [self._HEADER]
        for agent_id, data in self._data.items():
            lines.append(f"## Agent: {agent_id}")
            lines.append(f"*Last updated: {self._ts()}*\n")

            if data["turns"]:
                lines.append("### Conversation")
                lines.append("| Turn | Role | Content |")
                lines.append("|------|------|---------|")
                for t in data["turns"]:
                    content = t["content"].replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {t['ts']} | {t['role']} | {content} |")
                lines.append("")

            if data["masc_notes"]:
                lines.append("### MASC Notes")
                for note in data["masc_notes"]:
                    lines.append(f"- {note}")
                lines.append("")

            if data["sepo_events"]:
                lines.append("### SePO Events")
                for ev in data["sepo_events"]:
                    lines.append(f"- {ev}")
                lines.append("")

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError:
            pass  # Don't crash the agent pipeline on memory write failures

    def _load(self) -> None:
        """Parse existing markdown file back into _data."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return

        current_agent = None
        section = None
        table_started = False

        for line in content.splitlines():
            # Agent header
            m = re.match(r"^## Agent: (.+)$", line)
            if m:
                current_agent = m.group(1).strip()
                self._agent(current_agent)
                section = None
                table_started = False
                continue

            if current_agent is None:
                continue

            if line.startswith("### Conversation"):
                section = "turns"; table_started = False
            elif line.startswith("### MASC Notes"):
                section = "masc"; table_started = False
            elif line.startswith("### SePO Events"):
                section = "sepo"; table_started = False
            elif section == "turns":
                if line.startswith("| Turn"):
                    table_started = True
                elif line.startswith("|---"):
                    pass
                elif table_started and line.startswith("|"):
                    parts = [p.strip() for p in line.split("|")[1:-1]]
                    if len(parts) >= 3:
                        self._agent(current_agent)["turns"].append({
                            "ts":      parts[0],
                            "role":    parts[1],
                            "content": parts[2].replace("\\|", "|"),
                        })
            elif section == "masc" and line.startswith("- "):
                self._agent(current_agent)["masc_notes"].append(line[2:])
            elif section == "sepo" and line.startswith("- "):
                self._agent(current_agent)["sepo_events"].append(line[2:])
