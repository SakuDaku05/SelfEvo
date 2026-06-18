# -*- coding: utf-8 -*-
"""
examples/multi_agent_raw.py
============================
A multi-agent research pipeline built with google-genai.
NO validation framework. Raw. Fragile. Real-world messy.

Pipeline topology:
    User Query
        |
        v
    [TrendResearcher]  --> trends dict
        |
        v
    [MarketAnalyser]   --> market analysis dict
        |
        v
    [FactChecker]      --> fact-check dict
        |
        v
    [ReportWriter]     --> final text summary

Run:
    .venv\\Scripts\\python.exe examples/multi_agent_raw.py
"""
import os, sys, json, time

# ── make sure .env is loaded ────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL   = "gemini-2.5-flash"
client  = genai.Client(api_key=API_KEY)

# ── rate-limit helper ────────────────────────────────────────────────────
def call(prompt: str, system: str = "") -> str:
    """Raw Gemini call — no retry, no validation."""
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=0.7,
        max_output_tokens=512,
    )
    resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
    time.sleep(13)          # free-tier: 5 RPM = 1 call / 12s
    return resp.text or ""


# ======================================================================= #
# Agent 1 — TrendResearcher                                               #
# ======================================================================= #

def trend_researcher(topic: str) -> dict:
    """Research current trends for a given topic.
    Expected output: {trends, keywords, confidence_score, sources_count}
    """
    system = (
        "You are a market trend researcher. "
        "Reply ONLY with a raw JSON object — no markdown, no explanation. "
        "Keys: trends (list of strings), keywords (list of strings), "
        "confidence_score (float 0-1), sources_count (int)."
    )
    raw = call(f"Research the latest trends for: {topic}", system)
    # Try to parse — crash if the model didn't comply
    return json.loads(raw)


# ======================================================================= #
# Agent 2 — MarketAnalyser                                                #
# ======================================================================= #

def market_analyser(trends: dict) -> dict:
    """Analyse market impact from trend research.
    Expected output: {market_impact, impact_score, risk_level, opportunity_score}
    """
    system = (
        "You are a market analyst. "
        "Reply ONLY with a raw JSON object — no markdown, no explanation. "
        "Keys: market_impact (string), impact_score (float 0-10), "
        "risk_level (one of: low/medium/high/critical), "
        "opportunity_score (float 0-10)."
    )
    raw = call(f"Analyse market impact from these trend findings:\n{json.dumps(trends)}", system)
    return json.loads(raw)


# ======================================================================= #
# Agent 3 — FactChecker                                                   #
# ======================================================================= #

def fact_checker(trends: dict, analysis: dict) -> dict:
    """Cross-check claims in the trend + analysis outputs.
    Expected output: {verified_claims, disputed_claims, verdict, reliability_score}
    """
    system = (
        "You are a fact-checker. "
        "Reply ONLY with a raw JSON object — no markdown, no explanation. "
        "Keys: verified_claims (list of strings), disputed_claims (list of strings), "
        "verdict (one of: reliable/partially_reliable/unreliable), "
        "reliability_score (float 0-1)."
    )
    payload = {"trends": trends, "analysis": analysis}
    raw = call(f"Fact-check these research outputs:\n{json.dumps(payload)}", system)
    return json.loads(raw)


# ======================================================================= #
# Agent 4 — ReportWriter                                                  #
# ======================================================================= #

def report_writer(topic: str, trends: dict, analysis: dict, fact_check: dict) -> str:
    """Write an executive summary from all agent outputs.
    Expected output: plain text string (markdown OK)
    """
    system = (
        "You are an executive report writer. "
        "Write a concise 3-paragraph report. "
        "Use the data provided. Be professional and clear."
    )
    payload = {
        "topic": topic,
        "research": trends,
        "market_analysis": analysis,
        "fact_check": fact_check,
    }
    return call(
        f"Write an executive summary for this research:\n{json.dumps(payload)}",
        system,
    )


# ======================================================================= #
# Pipeline orchestrator                                                    #
# ======================================================================= #

def run_pipeline(topic: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Pipeline: {topic}")
    print(f"{'='*60}")

    # --- Step 1 ---
    print("\n[1/4] TrendResearcher ...", end=" ", flush=True)
    trends = trend_researcher(topic)
    print(f"OK  ({len(trends.get('trends', []))} trends found)")

    # --- Step 2 ---
    print("[2/4] MarketAnalyser ...", end=" ", flush=True)
    analysis = market_analyser(trends)
    print(f"OK  (impact_score={analysis.get('impact_score')}, risk={analysis.get('risk_level')})")

    # --- Step 3 ---
    print("[3/4] FactChecker ...", end=" ", flush=True)
    fact_check = fact_checker(trends, analysis)
    print(f"OK  (verdict={fact_check.get('verdict')}, reliability={fact_check.get('reliability_score')})")

    # --- Step 4 ---
    print("[4/4] ReportWriter ...", end=" ", flush=True)
    report = report_writer(topic, trends, analysis, fact_check)
    print("OK")

    return {
        "topic":      topic,
        "trends":     trends,
        "analysis":   analysis,
        "fact_check": fact_check,
        "report":     report,
    }


# ======================================================================= #
# Main — run two topics                                                    #
# ======================================================================= #

if __name__ == "__main__":
    topics = [
        "Generative AI in healthcare",
        "Electric vehicle adoption in emerging markets",
    ]

    results = []
    failures = []

    for topic in topics:
        try:
            result = run_pipeline(topic)
            results.append(result)
            print(f"\n--- REPORT PREVIEW ---")
            print(result["report"][:400])
            print("...")
        except Exception as e:
            failures.append({"topic": topic, "error": str(e)})
            print(f"\n[PIPELINE FAILED] {topic}: {e}")

    print(f"\n{'='*60}")
    print(f"  Done. {len(results)} succeeded, {len(failures)} failed.")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f['topic']}: {f['error']}")
    print(f"{'='*60}")
