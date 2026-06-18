# -*- coding: utf-8 -*-
"""
examples/multi_agent_simple.py
================================
The EXACT same 4-agent research pipeline as multi_agent_with_masc.py,
rewritten using the new one-liner API.

BEFORE (multi_agent_with_masc.py):
  ~40 lines of boilerplate per agent  →  ~160 lines just for 4 agents
  Subclassing BaseAgent, 4 @property methods, system_prompt setter,
  generate() with JSON parsing, etc.

AFTER (this file):
  1 call per agent with connector.add_llm_agent(...)  →  ~8 lines per agent
  No subclassing. No @property. No generate(). No JSON parsing.
  MASC/SePO still validates, corrects, evolves, and logs everything.

Run:
    .venv\\Scripts\\python.exe examples/multi_agent_simple.py
"""
import os, sys, json, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from connectors.agent_connector import AgentConnector
from evolution.llm_protocol import GeminiAdapter

# ── 1. Pick your LLM (swap this line for any other provider) ────────────
llm = GeminiAdapter(api_key=os.environ["GEMINI_API_KEY"], model="gemini-2.5-flash")

# ── 2. Create the connector ──────────────────────────────────────────────
connector = AgentConnector(
    llm_client=llm,
    log_path="logs/simple_pipeline.json",
    anomaly_threshold=2,
)

# ── 3. Register all 4 agents — one call each, no classes needed ─────────

connector.add_llm_agent(
    "trend_researcher",
    llm_client=llm,
    system=(
        "You are a market trend researcher. "
        "Reply ONLY with a valid JSON object — no markdown, no explanation. "
        "Required keys: trends (array of strings), keywords (array of strings), "
        "confidence_score (float 0-1), sources_count (integer >= 1)."
    ),
    user_template="Research the latest trends for: {query}",
    schema={
        "type": "object",
        "required": ["trends", "keywords", "confidence_score", "sources_count"],
        "properties": {
            "trends":           {"type": "array",   "items": {"type": "string"}, "minItems": 1},
            "keywords":         {"type": "array",   "items": {"type": "string"}, "minItems": 1},
            "confidence_score": {"type": "number",  "minimum": 0, "maximum": 1},
            "sources_count":    {"type": "integer", "minimum": 1},
        },
    },
    description="Researches current trends for a topic.",
)

connector.add_llm_agent(
    "market_analyser",
    llm_client=llm,
    system=(
        "You are a senior market analyst. "
        "Reply ONLY with a valid JSON object — no markdown, no explanation. "
        "Required keys: market_impact (string), impact_score (float 0-10), "
        "risk_level (low/medium/high/critical), opportunity_score (float 0-10)."
    ),
    user_template="Analyse market impact for these trends:\n{query}",
    schema={
        "type": "object",
        "required": ["market_impact", "impact_score", "risk_level", "opportunity_score"],
        "properties": {
            "market_impact":     {"type": "string"},
            "impact_score":      {"type": "number", "minimum": 0, "maximum": 10},
            "risk_level":        {"type": "string", "enum": ["low","medium","high","critical"]},
            "opportunity_score": {"type": "number", "minimum": 0, "maximum": 10},
        },
    },
    description="Analyses market impact from trend data.",
)

connector.add_llm_agent(
    "fact_checker",
    llm_client=llm,
    system=(
        "You are a rigorous fact-checker. "
        "Reply ONLY with a valid JSON object — no markdown, no explanation. "
        "Required keys: verified_claims (array), disputed_claims (array), "
        "verdict (reliable/partially_reliable/unreliable), reliability_score (float 0-1)."
    ),
    user_template="Fact-check this research:\n{query}",
    schema={
        "type": "object",
        "required": ["verified_claims", "disputed_claims", "verdict", "reliability_score"],
        "properties": {
            "verified_claims":   {"type": "array",  "items": {"type": "string"}},
            "disputed_claims":   {"type": "array",  "items": {"type": "string"}},
            "verdict":           {"type": "string", "enum": ["reliable","partially_reliable","unreliable"]},
            "reliability_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
    description="Cross-checks claims in research outputs.",
)

connector.add_llm_agent(
    "report_writer",
    llm_client=llm,
    system=(
        "You are a professional executive report writer. "
        "Write a concise 3-paragraph business report. "
        "Be specific, factual, and clear. Flowing prose only."
    ),
    user_template="Write an executive report based on:\n{query}",
    # No schema → MASC just checks it's non-empty
    description="Writes the final executive summary.",
)

print("Registered agents:", [a["agent_id"] for a in connector.list_agents()])

# ── 4. Retry wrapper (free-tier rate limit safety) ───────────────────────

def safe_run(agent_id: str, query: str) -> dict:
    """Run with retry on 429/503 from within the agent call."""
    for attempt in range(3):
        try:
            return connector.run(agent_id, query)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "503" in msg or "EXHAUSTED" in msg or "UNAVAILABLE" in msg:
                wait = 65 if "429" in msg else 20
                print(f"    [retry {attempt+1}/3] {exc.__class__.__name__} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    return connector.run(agent_id, query)   # final attempt


# ── 5. The pipeline — identical logic, much cleaner ──────────────────────

def run_pipeline(topic: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Pipeline: {topic}")
    print(f"{'='*60}")
    corrections = []

    def step(name, query, label):
        print(f"  {label} ...", end=" ", flush=True)
        r = safe_run(name, query)
        if r["corrected"]:
            tag = (r["anomaly"] or {}).get("type", "?")
            corrections.append(tag)
            print(f"CORRECTED [{tag}]")
        elif r["error"]:
            print(f"ERROR: {r['error'][:60]}")
        else:
            print(f"OK  ({r['latency_ms']:.0f} ms)")
        return r["output"]

    trends     = step("trend_researcher", topic,                          "[1/4] TrendResearcher")
    analysis   = step("market_analyser",  json.dumps(trends, default=str),"[2/4] MarketAnalyser ")
    fact_check = step("fact_checker",
                      json.dumps({"trends": trends, "analysis": analysis}, default=str),
                      "[3/4] FactChecker    ")
    report     = step("report_writer",
                      json.dumps({"topic": topic, "research": trends,
                                  "analysis": analysis, "fact_check": fact_check}, default=str),
                      "[4/4] ReportWriter   ")

    return {"topic": topic, "trends": trends, "analysis": analysis,
            "fact_check": fact_check, "report": report, "corrections": corrections}


# ── 6. Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    topics = [
        "Generative AI in healthcare",
        "Electric vehicle adoption in emerging markets",
    ]

    results = []
    for topic in topics:
        r = run_pipeline(topic)
        results.append(r)
        rpt = str(r["report"])
        print(f"\n  Report: {rpt[:300]}{'...' if len(rpt) > 300 else ''}")
        if r["corrections"]:
            print(f"  MASC corrections: {r['corrections']}")

    # ── Stats ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MASC/SePO Stats  (simple API version)")
    print(f"{'='*60}")
    g = connector.stats()["global"]
    print(f"  Agent calls   : {g['total_runs']}")
    print(f"  Corrections   : {g['correction_count']}  ({g['correction_rate']*100:.1f}%)")
    print(f"  Errors        : {g['error_count']}")
    print(f"  Avg latency   : {g['avg_latency_ms']:.0f} ms")
    print(f"\n  {len(results)}/2 pipelines completed  |  raw version: 0/2")
    print(f"  View dashboard: streamlit run dashboard/app.py")
    print(f"{'='*60}")
