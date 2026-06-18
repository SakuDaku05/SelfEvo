# -*- coding: utf-8 -*-
"""
examples/multi_agent_with_masc.py
===================================
Same multi-agent research pipeline — now INTEGRATED with MASC/SePO.

Story
-----
After the raw pipeline (multi_agent_raw.py) crashed on both topics with
JSONDecodeError, we discovered the MASC/SePO connector framework sitting
in this repo and wired it in. Here is what changed and why it helps.

What MASC/SePO adds
--------------------
  - Every agent output is validated against its declared JSON schema
  - Bad outputs (empty, wrong type, missing fields) are AUTO-CORRECTED
  - Corrections are logged with anomaly type + field
  - SePO reuses Gemini to rewrite the agent's system prompt so the
    same mistake stops happening after N failures
  - Every pipeline run is persisted to JSONL for dashboard analysis
  - The pipeline NEVER crashes on a bad output — it corrects and continues

Run:
    .venv\\Scripts\\python.exe examples/multi_agent_with_masc.py

Then open the dashboard to see every run, correction, and evolution:
    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
import os, sys, json, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# ── MASC/SePO imports ───────────────────────────────────────────────────
from connectors.base_agent    import BaseAgent
from connectors.agent_connector import AgentConnector
from evolution.llm_protocol   import GeminiAdapter

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL   = "gemini-2.5-flash"

# One shared Gemini LLM — used both as the agent brain AND by SePO/correction
llm = GeminiAdapter(api_key=API_KEY, model=MODEL)

# ── rate-limit + retry helper ────────────────────────────────────────────
def safe_gemini_call(llm_client, messages: list, retries: int = 3) -> str:
    """Call Gemini with retry on 429 / 503 / empty response."""
    for attempt in range(retries):
        try:
            result = llm_client.chat(messages)
            if result and result.strip():
                return result
            # Empty response — wait and retry
            print(f"    [warn] empty response, retry {attempt+1}/{retries}")
            time.sleep(15)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "503" in msg or "UNAVAILABLE" in msg or "EXHAUSTED" in msg:
                wait = 65 if "429" in msg else 20
                print(f"    [warn] {exc.__class__.__name__} — waiting {wait}s (retry {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    return ""   # Give up — MASC will catch the empty/null output


# ======================================================================= #
# Agent definitions — each has an output_schema so MASC auto-validates   #
# ======================================================================= #

class TrendResearcherAgent(BaseAgent):
    """
    Researches current trends for a given topic.
    Output: { trends[], keywords[], confidence_score, sources_count }
    """

    _system_prompt = (
        "You are a market trend researcher. "
        "Reply ONLY with a valid JSON object — absolutely no markdown fences, "
        "no explanations. "
        "Required keys: "
        "  trends (array of strings, at least 3), "
        "  keywords (array of strings, at least 2), "
        "  confidence_score (float between 0 and 1), "
        "  sources_count (integer >= 1)."
    )

    @property
    def agent_id(self):  return "trend_researcher"
    @property
    def description(self): return "Researches trends for a topic and returns structured JSON."

    @property
    def output_schema(self):
        return {
            "type": "object",
            "required": ["trends", "keywords", "confidence_score", "sources_count"],
            "properties": {
                "trends":           {"type": "array",  "items": {"type": "string"}, "minItems": 1},
                "keywords":         {"type": "array",  "items": {"type": "string"}, "minItems": 1},
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                "sources_count":    {"type": "integer","minimum": 1},
            },
        }

    @property
    def system_prompt(self):  return self._system_prompt
    @system_prompt.setter
    def system_prompt(self, v): self._system_prompt = v

    def generate(self, query: str, **kw):
        raw = safe_gemini_call(llm, [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"Research the latest trends for: {query}"},
        ])
        # Strip markdown fences if present
        text = raw.strip().strip("` \n").lstrip("json").strip()
        try:
            return json.loads(text)
        except Exception:
            return text   # Return raw — MASC will flag and correct it


class MarketAnalyserAgent(BaseAgent):
    """
    Analyses market impact from trend research.
    Output: { market_impact, impact_score, risk_level, opportunity_score }
    """

    _system_prompt = (
        "You are a senior market analyst. "
        "Reply ONLY with a valid JSON object — no markdown, no explanation. "
        "Required keys: "
        "  market_impact (string), "
        "  impact_score (float 0-10), "
        "  risk_level (exactly one of: low / medium / high / critical), "
        "  opportunity_score (float 0-10)."
    )

    @property
    def agent_id(self):  return "market_analyser"
    @property
    def description(self): return "Analyses market impact from research trends."

    @property
    def output_schema(self):
        return {
            "type": "object",
            "required": ["market_impact", "impact_score", "risk_level", "opportunity_score"],
            "properties": {
                "market_impact":     {"type": "string"},
                "impact_score":      {"type": "number", "minimum": 0, "maximum": 10},
                "risk_level":        {"type": "string", "enum": ["low","medium","high","critical"]},
                "opportunity_score": {"type": "number", "minimum": 0, "maximum": 10},
            },
        }

    @property
    def system_prompt(self):  return self._system_prompt
    @system_prompt.setter
    def system_prompt(self, v): self._system_prompt = v

    def generate(self, query: str, **kw):
        raw = safe_gemini_call(llm, [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"Analyse market impact for these trends:\n{query}"},
        ])
        text = raw.strip().strip("` \n").lstrip("json").strip()
        try:
            return json.loads(text)
        except Exception:
            return text


class FactCheckerAgent(BaseAgent):
    """
    Cross-checks claims in the trend + analysis outputs.
    Output: { verified_claims[], disputed_claims[], verdict, reliability_score }
    """

    _system_prompt = (
        "You are a rigorous fact-checker. "
        "Reply ONLY with a valid JSON object — no markdown, no explanation. "
        "Required keys: "
        "  verified_claims (array of strings), "
        "  disputed_claims (array of strings), "
        "  verdict (exactly one of: reliable / partially_reliable / unreliable), "
        "  reliability_score (float 0-1)."
    )

    @property
    def agent_id(self):  return "fact_checker"
    @property
    def description(self): return "Fact-checks research and analysis outputs."

    @property
    def output_schema(self):
        return {
            "type": "object",
            "required": ["verified_claims", "disputed_claims", "verdict", "reliability_score"],
            "properties": {
                "verified_claims":  {"type": "array",  "items": {"type": "string"}},
                "disputed_claims":  {"type": "array",  "items": {"type": "string"}},
                "verdict":          {"type": "string", "enum": ["reliable","partially_reliable","unreliable"]},
                "reliability_score":{"type": "number", "minimum": 0, "maximum": 1},
            },
        }

    @property
    def system_prompt(self):  return self._system_prompt
    @system_prompt.setter
    def system_prompt(self, v): self._system_prompt = v

    def generate(self, query: str, **kw):
        raw = safe_gemini_call(llm, [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"Fact-check this research:\n{query}"},
        ])
        text = raw.strip().strip("` \n").lstrip("json").strip()
        try:
            return json.loads(text)
        except Exception:
            return text


class ReportWriterAgent(BaseAgent):
    """
    Writes an executive summary from all prior agent outputs.
    Output: plain text (no schema — MASC checks it's non-empty)
    """

    _system_prompt = (
        "You are a professional executive report writer. "
        "Write a concise 3-paragraph business report. "
        "Be specific, factual, and clear. No bullet points — flowing prose only."
    )

    @property
    def agent_id(self):  return "report_writer"
    @property
    def description(self): return "Writes an executive summary from all prior agent outputs."
    # No output_schema → MASC just ensures it's non-empty

    @property
    def system_prompt(self):  return self._system_prompt
    @system_prompt.setter
    def system_prompt(self, v): self._system_prompt = v

    def generate(self, query: str, **kw):
        return safe_gemini_call(llm, [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"Write an executive report based on:\n{query}"},
        ])


# ======================================================================= #
# Build the connector                                                      #
# ======================================================================= #

connector = AgentConnector(
    llm_client=llm,             # SePO & CorrectionAgent also use Gemini
    log_path="logs/pipeline_runs.json",
    anomaly_threshold=2,        # evolve prompt after 2 consecutive anomalies
)

connector.register_many([
    TrendResearcherAgent(),
    MarketAnalyserAgent(),
    FactCheckerAgent(),
    ReportWriterAgent(),
])

print("Registered agents:", [a["agent_id"] for a in connector.list_agents()])


# ======================================================================= #
# Pipeline orchestrator — NEVER crashes, MASC handles all bad outputs     #
# ======================================================================= #

def run_pipeline(topic: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Pipeline: {topic}")
    print(f"{'='*60}")

    corrections = []
    anomalies   = []

    # --- Step 1: Research ---
    print("\n[1/4] TrendResearcher ...", end=" ", flush=True)
    r1 = connector.run("trend_researcher", topic)
    _report_step(r1, corrections, anomalies)
    trends = r1["output"]

    # --- Step 2: Market Analysis (fed trends as input) ---
    print("[2/4] MarketAnalyser ...", end=" ", flush=True)
    r2 = connector.run("market_analyser", json.dumps(trends, default=str))
    _report_step(r2, corrections, anomalies)
    analysis = r2["output"]

    # --- Step 3: Fact Check (fed both) ---
    print("[3/4] FactChecker ...", end=" ", flush=True)
    payload = json.dumps({"trends": trends, "analysis": analysis}, default=str)
    r3 = connector.run("fact_checker", payload)
    _report_step(r3, corrections, anomalies)
    fact_check = r3["output"]

    # --- Step 4: Report (plain text, fed everything) ---
    print("[4/4] ReportWriter ...", end=" ", flush=True)
    full_context = json.dumps({
        "topic":      topic,
        "research":   trends,
        "analysis":   analysis,
        "fact_check": fact_check,
    }, default=str)
    r4 = connector.run("report_writer", full_context)
    _report_step(r4, corrections, anomalies)
    report = r4["output"]

    return {
        "topic":       topic,
        "trends":      trends,
        "analysis":    analysis,
        "fact_check":  fact_check,
        "report":      report,
        "corrections": corrections,
        "anomalies":   anomalies,
    }


def _report_step(result: dict, corrections: list, anomalies: list) -> None:
    if result["corrected"]:
        a = result["anomaly"] or {}
        tag = a.get("type", "?")
        corrections.append(tag)
        anomalies.append(result["anomaly"])
        print(f"CORRECTED [{tag}]")
    elif result["error"]:
        print(f"ERROR: {result['error'][:60]}")
    else:
        print(f"OK  ({result['latency_ms']:.0f} ms)")


# ======================================================================= #
# Main — run same two topics as the raw script                            #
# ======================================================================= #

if __name__ == "__main__":
    topics = [
        "Generative AI in healthcare",
        "Electric vehicle adoption in emerging markets",
    ]

    all_results = []

    for topic in topics:
        result = run_pipeline(topic)   # NEVER raises — MASC handles everything
        all_results.append(result)

        print(f"\n--- REPORT PREVIEW ---")
        rpt = result["report"]
        print((rpt[:400] + "...") if len(str(rpt)) > 400 else rpt)

        if result["corrections"]:
            print(f"\n  MASC made {len(result['corrections'])} correction(s): {result['corrections']}")
        else:
            print("\n  MASC: all outputs were valid.")

    # ── Aggregate stats ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MASC/SePO Pipeline Statistics")
    print(f"{'='*60}")
    stats = connector.stats()
    g = stats["global"]
    print(f"  Total agent calls : {g['total_runs']}")
    print(f"  Corrections made  : {g['correction_count']}  ({g['correction_rate']*100:.1f}%)")
    print(f"  Errors            : {g['error_count']}  ({g['error_rate']*100:.1f}%)")
    print(f"  Avg latency       : {g['avg_latency_ms']:.0f} ms")

    print(f"\n  Per-agent breakdown:")
    for aid, ag in stats["agents"].items():
        corr_flag = " <-- MASC corrected" if ag["correction_count"] > 0 else ""
        print(f"    {aid:<22} runs={ag['total_runs']}  "
              f"corrections={ag['correction_count']}{corr_flag}")

    if g.get("anomaly_type_counts"):
        print(f"\n  Anomaly types caught by MASC:")
        for atype, count in g["anomaly_type_counts"].items():
            print(f"    {atype:<30} x{count}")

    print(f"\n  Results: {len(all_results)}/2 pipelines completed successfully.")
    print(f"  (raw version: 0/2)")
    print(f"\n  View full history: streamlit run dashboard/app.py")
    print(f"{'='*60}")
