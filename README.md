# 🧠 SelfEvo — Self-Evolving Agent Connector Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)

A **LLM-agnostic** connector framework that adds automatic validation, correction, and self-evolving prompts to **any** agent pipeline — with zero boilerplate.

| Component | Role |
|-----------|------|
| **MASC** | Multi-Aspect Schema Check — auto-validates outputs, detects anomalies, corrects them |
| **SePO** | Self-Evolving Prompt Optimizer — rewrites agent prompts to prevent repeated failures |
| **Connector** | Routes queries, wires MASC + SePO together, logs every run |
| **Dashboard** | Streamlit UI — live KPIs, correction history, evolution timeline |
| **API** | FastAPI server — 9 endpoints for querying, stats, and observability |

> **No vendor lock-in.** Works with OpenAI, Anthropic, Gemini, Ollama, Cohere, Mistral, Azure — or your own provider. MASC corrections work with **no LLM at all**.

---

## ⚡ Quickstart

```bash
git clone https://github.com/SakuDaku05/SelfEvo.git
cd SelfEvo
pip install -r requirements.txt
python examples/quickstart.py
```

---

## 🔌 Step 1 — Pick your LLM (one line)

```python
# Google Gemini (free tier)
from evolution.llm_protocol import GeminiAdapter
llm = GeminiAdapter(api_key="AIza…", model="gemini-2.5-flash")

# OpenAI
from evolution.llm_protocol import OpenAIAdapter
llm = OpenAIAdapter(api_key="sk-…", model="gpt-4o")

# Anthropic
from evolution.llm_protocol import AnthropicAdapter
llm = AnthropicAdapter(api_key="sk-ant-…")

# Ollama (local, free — no key needed)
from evolution.llm_protocol import OllamaAdapter
llm = OllamaAdapter(model="llama3")

# Cohere / Mistral / Azure — same pattern
# Any custom provider — just expose .chat(messages) -> str
class MyLLM:
    def chat(self, messages: list[dict]) -> str:
        return my_provider.call(messages)
```

---

## 🏗️ Step 2 — Connect your agents (one line each)

### Option A — wrap any function (simplest)

```python
from connectors.agent_connector import AgentConnector

connector = AgentConnector(llm_client=llm)

connector.add_fn(
    "my_agent",
    fn=lambda query, **kw: my_existing_function(query),
    system="You are a helpful assistant. Reply in JSON.",
    schema={
        "type": "object",
        "required": ["answer", "confidence"],
        "properties": {
            "answer":     {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
)
```

### Option B — LLM agent in one call (no class needed)

```python
connector.add_llm_agent(
    "summariser",
    llm_client=llm,
    system="Summarise the input in 3 sentences. Reply in JSON with key 'summary'.",
    user_template="Summarise this:\n{query}",
    schema={
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    },
)
```

### Option C — chain multiple agents

```python
connector \
    .add_llm_agent("researcher", llm_client=llm, system="...", schema={...}) \
    .add_llm_agent("analyser",   llm_client=llm, system="...", schema={...}) \
    .add_llm_agent("writer",     llm_client=llm, system="...")
```

---

## 🚀 Step 3 — Run (never crashes, MASC handles everything)

```python
result = connector.run("my_agent", "What is the meaning of life?")

print(result["output"])     # validated (and corrected if needed) output
print(result["corrected"])  # True if MASC had to fix something
print(result["anomaly"])    # anomaly type, or None
print(result["latency_ms"]) # round-trip time
```

---

## 🛡️ MASC — What gets validated automatically

Declare an `output_schema` (JSON Schema) and MASC auto-derives all these rules:

| Rule | What it catches |
|------|----------------|
| `null_output` | None / empty response |
| `type_mismatch` | Wrong top-level type (string instead of object, etc.) |
| `required_fields` | Missing required keys |
| `property_types` | Field values of wrong type |
| `numeric_range` | Numbers outside `minimum`/`maximum` |
| `string_pattern` | Strings not matching `pattern` regex |
| `enum_violation` | Enum fields with undeclared values |
| `empty_array` | Arrays not meeting `minItems` |
| `string_not_empty` | Blank plain-text output |

**No schema?** MASC still validates: output is non-null and non-empty.

### Add a custom rule (plug-in system)

```python
from interceptor.masc_validator import MASCValidator, ValidationRule

class ToxicityFilter(ValidationRule):
    name = "toxicity_filter"

    def check(self, output, schema):
        if isinstance(output, str) and "badword" in output.lower():
            return {"type": "toxicity_filter", "detail": "Banned term",
                    "field": None, "value": output}

connector.validator.add_rule(ToxicityFilter())
```

---

## 🧬 SePO — Self-Healing Prompts

After `anomaly_threshold` consecutive bad outputs on the same agent:

1. **With LLM** → SePO sends the current prompt + anomaly details to your LLM and gets back a rewritten prompt.
2. **Without LLM** → SePO appends a targeted instruction patch (e.g. *"CRITICAL: Always return a non-empty JSON object."*).
3. The new prompt is saved to `logs/evolution_history.jsonl` and applied automatically.

```python
# Set how many anomalies trigger an evolution cycle (default: 3)
connector = AgentConnector(llm_client=llm, anomaly_threshold=2)
```

---

## 📊 Observability

### Stats

```python
stats = connector.stats()
print(stats["global"]["correction_rate"])       # e.g. 0.23
print(stats["agents"]["my_agent"]["total_runs"])
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

5 pages: **Overview KPIs · Agent Drill-Down · Evolution History · Live Logs · MASC Rules**

### API Server

```bash
uvicorn api.server:app --reload
# → http://localhost:8000/docs
```

---

## 🔬 Advanced — subclass BaseAgent for full control

For agents that need custom lifecycle hooks (`on_correction`, `on_evolution`), use the full class API:

```python
from connectors.base_agent import BaseAgent

class MyAgent(BaseAgent):
    @property
    def agent_id(self) -> str: return "my_agent"
    @property
    def output_schema(self): return {"type": "object", ...}

    def generate(self, query: str, **kwargs):
        return my_logic(query)

    def on_evolution(self, new_prompt: str):
        # Called whenever SePO rewrites the prompt
        self.save_prompt_to_db(new_prompt)

connector.register(MyAgent())
```

> The `add_fn` / `add_llm_agent` one-liners use this internally — subclassing is only needed for advanced hooks.

---

## 📁 Project Structure

```
SelfEvo/
├── connectors/
│   ├── base_agent.py          # Abstract agent interface (advanced use)
│   ├── quick_agent.py         # FunctionAgent — wraps any callable
│   └── agent_connector.py     # Orchestrator: add_fn, add_llm_agent, run
├── interceptor/
│   ├── masc_validator.py      # 9 schema-driven validation rules + plugin API
│   └── correction_agent.py    # Heuristic + LLM-assisted correction
├── evolution/
│   ├── llm_protocol.py        # 7 LLM provider adapters + protocol
│   ├── sepo_engine.py         # Prompt evolution engine
│   └── evolution_tracker.py   # JSONL history analytics
├── logs/
│   └── logger.py              # Thread-safe JSONL logger + aggregate stats
├── api/
│   └── server.py              # FastAPI (9 endpoints)
├── dashboard/
│   └── app.py                 # Streamlit dashboard (5 pages)
├── examples/
│   ├── quickstart.py          # Full demo, no API key needed
│   ├── multi_agent_simple.py  # 4-agent pipeline using the simple API
│   └── agents.py              # Finance, Health, Legal, Sentiment, Echo agents
└── tests/
    ├── test_masc_validator.py  # 38 MASC rule tests (offline)
    ├── test_correction_agent.py# 22 correction tests (offline)
    ├── test_agent_connector.py # 17 connector + SePO tests (offline)
    └── test_quick_api.py       # 8 add_fn / add_llm_agent tests (offline)
```

---

## 📦 Install

```bash
pip install -r requirements.txt

# Optional LLM SDKs
pip install google-genai          # Gemini
pip install openai                # OpenAI / Azure
pip install anthropic             # Anthropic
pip install ollama                # Ollama (local)
pip install cohere                # Cohere
pip install mistralai             # Mistral
```

---

## 🧪 Tests

```bash
# All offline tests (no API key needed)
pytest tests/ -v

# With live Gemini tests
GEMINI_API_KEY=AIza… pytest tests/ -v
```

---

## License

MIT © 2025 — PRs welcome! ⭐ Star the repo if this saves you from a JSONDecodeError at 2am.
