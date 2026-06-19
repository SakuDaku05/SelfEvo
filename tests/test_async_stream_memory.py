"""
tests/test_async_stream_memory.py
Offline tests for async, streaming, and markdown memory features.
No API key needed â€” uses mock agents.
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.agent_connector import AgentConnector
from connectors.memory import MarkdownMemory


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def make_connector(tmp_path, memory=False):
    c = AgentConnector(log_path=os.path.join(tmp_path, "logs.json"))
    c.add_fn("echo",   lambda q, **kw: q,           description="echo")
    c.add_fn("json_a", lambda q, **kw: {"value": 42},
             schema={"type":"object","required":["value"],"properties":{"value":{"type":"number"}}})
    c.add_fn("bad",    lambda q, **kw: "not json",
             schema={"type":"object","required":["x"],"properties":{"x":{"type":"number"}}})
    if memory:
        c.use_memory(path=os.path.join(tmp_path, "memory.md"))
    return c


# â”€â”€ Async tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_arun_sync_agent():
    """arun() works on a regular sync agent (runs in executor)."""
    with tempfile.TemporaryDirectory() as tmp:
        c = make_connector(tmp)
        result = asyncio.run(c.arun("echo", "hello async"))
        assert result["output"] == "hello async"
        assert result["error"] is None
    print("[PASS] arun() with sync agent")


def test_arun_async_agent():
    """arun() works on a native async agent function."""
    with tempfile.TemporaryDirectory() as tmp:
        c = AgentConnector(log_path=os.path.join(tmp, "logs.json"))

        async def async_fn(q, **kw):
            await asyncio.sleep(0)   # simulate async I/O
            return {"result": q.upper()}

        c.add_fn("async_upper", async_fn,
                 schema={"type":"object","required":["result"],"properties":{"result":{"type":"string"}}})
        result = asyncio.run(c.arun("async_upper", "hello"))
        assert result["output"]["result"] == "HELLO"
        assert result["corrected"] is False
    print("[PASS] arun() with native async agent")


def test_arun_masc_correction():
    """arun() still runs MASC correction on bad async output."""
    with tempfile.TemporaryDirectory() as tmp:
        c = make_connector(tmp)
        result = asyncio.run(c.arun("bad", "test"))
        assert result["corrected"] is True
        assert result["anomaly"]["type"] == "type_mismatch"
        assert isinstance(result["output"], dict)
    print("[PASS] arun() â€” MASC correction works on async path")


def test_parallel_arun():
    """asyncio.gather() runs agents in parallel."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            c = make_connector(tmp)
            results = await asyncio.gather(
                c.arun("echo", "q1"),
                c.arun("echo", "q2"),
                c.arun("echo", "q3"),
            )
            return results

    results = asyncio.run(run())
    outputs = {r["output"] for r in results}
    assert outputs == {"q1", "q2", "q3"}
    print("[PASS] parallel arun() via asyncio.gather()")


# â”€â”€ Streaming tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_stream_yields_chunks_then_done():
    """stream() yields at least one chunk and a final done event."""
    with tempfile.TemporaryDirectory() as tmp:
        c = make_connector(tmp)
        events = list(c.stream("echo", "hello stream"))
        chunk_events = [e for e in events if e["event"] == "chunk"]
        done_events  = [e for e in events if e["event"] == "done"]
        assert len(chunk_events) >= 1
        assert len(done_events) == 1
        assert done_events[0]["output"] == "hello stream"
    print("[PASS] stream() yields chunks + done event")


def test_stream_masc_correction_on_done():
    """stream() applies MASC on the assembled output at done event."""
    with tempfile.TemporaryDirectory() as tmp:
        c = make_connector(tmp)
        events = list(c.stream("bad", "test"))
        done = next(e for e in events if e["event"] == "done")
        assert done["corrected"] is True
        assert isinstance(done["output"], dict)
    print("[PASS] stream() â€” MASC correction on assembled output")


def test_astream_async():
    """astream() works as async generator."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            c = make_connector(tmp)
            events = []
            async for event in c.astream("echo", "async stream test"):
                events.append(event)
            return events

    events = asyncio.run(run())
    assert any(e["event"] == "chunk" for e in events)
    done = next(e for e in events if e["event"] == "done")
    assert done["output"] == "async stream test"
    print("[PASS] astream() async generator")


def test_stream_custom_generator_agent():
    """stream() works with an agent that overrides stream_generate()."""
    from connectors.base_agent import BaseAgent

    class ChunkyAgent(BaseAgent):
        @property
        def agent_id(self): return "chunky"
        def generate(self, query, **kw): return "".join(["Hello", " ", "World"])
        def stream_generate(self, query, **kw):
            for word in ["Hello", " ", "World"]:
                yield word

    with tempfile.TemporaryDirectory() as tmp:
        c = AgentConnector(log_path=os.path.join(tmp, "logs.json"))
        c.register(ChunkyAgent())
        events = list(c.stream("chunky", "anything"))
        chunks = [e["text"] for e in events if e["event"] == "chunk"]
        assert chunks == ["Hello", " ", "World"]
        done = next(e for e in events if e["event"] == "done")
        assert done["output"] == "Hello World"
    print("[PASS] stream() with custom stream_generate() override")


# â”€â”€ Memory tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_memory_records_turns(tmp_path):
    """Memory file records user + agent turns after run()."""
    mem_path = str(tmp_path / "memory.md")
    c = AgentConnector(log_path=str(tmp_path / "logs.json"))
    c.add_fn("echo", lambda q, **kw: q)
    c.use_memory(path=mem_path)

    c.run("echo", "first question")
    c.run("echo", "second question")

    mem = MarkdownMemory(path=mem_path)
    summary = mem.summary()
    assert "echo" in summary
    assert summary["echo"]["turns"] == 4   # 2 user + 2 agent
    print("[PASS] memory records turns after run()")


def test_memory_context_injected(tmp_path):
    """Memory context is prepended to the query on subsequent calls."""
    mem_path = str(tmp_path / "memory.md")
    received_queries = []

    def capture_fn(q, **kw):
        received_queries.append(q)
        return q

    c = AgentConnector(log_path=str(tmp_path / "logs.json"))
    c.add_fn("capture", capture_fn)
    c.use_memory(path=mem_path, inject_turns=5)

    c.run("capture", "first")   # turn 1 â€” no prior context
    c.run("capture", "second")  # turn 2 â€” should have context from turn 1

    # First query should be plain (no context yet)
    assert received_queries[0] == "first"
    # Second query should have memory context injected
    assert "first" in received_queries[1] or "Current query: second" in received_queries[1]
    print("[PASS] memory context injected into subsequent queries")


def test_memory_masc_notes(tmp_path):
    """MASC anomalies are recorded as notes in memory."""
    mem_path = str(tmp_path / "memory.md")
    c = AgentConnector(log_path=str(tmp_path / "logs.json"))
    c.add_fn("bad", lambda q, **kw: "not json",
             schema={"type":"object","required":["x"],"properties":{"x":{"type":"number"}}})
    c.use_memory(path=mem_path)
    c.run("bad", "test")

    mem = MarkdownMemory(path=mem_path)
    assert mem.summary()["bad"]["masc_notes"] >= 1
    print("[PASS] MASC anomaly recorded as note in memory")


def test_memory_file_round_trips(tmp_path):
    """Memory file can be saved and reloaded correctly."""
    mem_path = str(tmp_path / "memory.md")
    mem1 = MarkdownMemory(path=mem_path)
    mem1.add_turn("agent_x", "user",  "hello")
    mem1.add_turn("agent_x", "agent", "world")
    mem1.add_masc_note("agent_x", "null_output", "was empty")
    mem1.add_sepo_event("agent_x", "null_output", method="heuristic")

    # Reload from disk
    mem2 = MarkdownMemory(path=mem_path)
    assert mem2.summary()["agent_x"]["turns"] == 2
    assert mem2.summary()["agent_x"]["masc_notes"] == 1
    assert mem2.summary()["agent_x"]["sepo_events"] == 1
    print("[PASS] memory file round-trips (save -> reload)")


def test_memory_clear_agent(tmp_path):
    """clear_agent() wipes only that agent's memory."""
    mem = MarkdownMemory(path=str(tmp_path / "memory.md"))
    mem.add_turn("a1", "user", "q1")
    mem.add_turn("a2", "user", "q2")
    mem.clear_agent("a1")
    assert "a1" not in mem.summary()
    assert "a2" in mem.summary()
    print("[PASS] clear_agent() wipes only target agent")


def test_memory_arun_records_turns(tmp_path):
    """arun() also records turns in memory."""
    mem_path = str(tmp_path / "memory.md")
    c = AgentConnector(log_path=str(tmp_path / "logs.json"))
    c.add_fn("echo", lambda q, **kw: q)
    c.use_memory(path=mem_path)
    asyncio.run(c.arun("echo", "async memory test"))
    mem = MarkdownMemory(path=mem_path)
    assert mem.summary()["echo"]["turns"] == 2   # 1 user + 1 agent
    print("[PASS] arun() records turns in memory")


# â”€â”€ Runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())

    test_arun_sync_agent()
    test_arun_async_agent()
    test_arun_masc_correction()
    test_parallel_arun()
    test_stream_yields_chunks_then_done()
    test_stream_masc_correction_on_done()
    test_astream_async()
    test_stream_custom_generator_agent()
    test_memory_records_turns(tmp / "mem1")
    test_memory_context_injected(tmp / "mem2")
    test_memory_masc_notes(tmp / "mem3")
    test_memory_file_round_trips(tmp / "mem4")
    test_memory_clear_agent(tmp / "mem5")
    test_memory_arun_records_turns(tmp / "mem6")

    print("\n All async + stream + memory tests PASSED.")
