# 🧠 MASC / SePO — Self-Evolving Agent Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)

A **LLM-agnostic** connector framework where:

| Component | Role |
|-----------|------|
| **Agents** | Any domain logic wrapped in a single `generate()` method |
| **MASC** | Multi-Aspect Schema Check — validates outputs against schemas, detects anomalies, applies corrections |
| **SePO** | Self-Evolving Prompt Optimizer — learns from repeated anomalies and rewrites agent prompts |
| **Logger** | Structured JSONL logs with per-agent stats |
| **API** | FastAPI server — send queries, get validated + evolved responses |
| **Dashboard** | Streamlit dashboard — live KPIs, corrections, evolution history |

> **No LLM vendor lock-in.** Bring OpenAI, Anthropic, Gemini, Ollama, Cohere, Mistral, Azure, or your own provider. MASC validation and SePO heuristics work even with **no LLM at all**.

---

## ⚡ Quickstart (no API key needed)

```bash
git clone https://github.com/your-org/masc-sepo
cd masc-sepo
pip install -r requirements.txt
python examples/quickstart.py
```

You'll see:
- Normal agents running and returning validated outputs
- A broken agent triggering MASC corrections
- SePO evolving the broken agent's prompt after 2 consecutive anomalies
- Aggregate stats

---

## 🔌 Connect Your Own LLM

Choose your provider (install the relevant SDK first):

```python
# OpenAI
from evolution.llm_protocol import OpenAIAdapter
llm = OpenAIAdapter(api_key="sk-…", model="gpt-4o")

# Anthropic
from evolution.llm_protocol import AnthropicAdapter
llm = AnthropicAdapter(api_key="sk-ant-…")

# Google Gemini
from evolution.llm_protocol import GeminiAdapter
llm = GeminiAdapter(api_key="AIza…", model="gemini-2.5-flash")

# Ollama (local, free)
from evolution.llm_protocol import OllamaAdapter
llm = OllamaAdapter(model="llama3")      # needs Ollama running

# Cohere
from evolution.llm_protocol import CohereAdapter
llm = CohereAdapter(api_key="…")

# Mistral
from evolution.llm_protocol import MistralAdapter
llm = MistralAdapter(api_key="…")

# Azure OpenAI
from evolution.llm_protocol import AzureOpenAIAdapter
llm = AzureOpenAIAdapter(api_key="…", azure_endpoint="…", api_version="…", deployment_name="…")

# Any custom provider
class MyLLM:
    def chat(self, messages: list[dict]) -> str:
        return my_provider.call(messages)

llm = MyLLM()
```

Pass it in:

```python
from connectors.agent_connector import AgentConnector
connector = AgentConnector(llm_client=llm)
```

---

## 🏗️ Write Your Own Agent

```python
from connectors.base_agent import BaseAgent
from typing import Any, Dict

class MyAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return "my_agent"

    @property
    def description(self) -> str:
        return "Does something amazing."

    @property
    def output_schema(self) -> Dict[str, Any]:
        # Declare this and MASC auto-validates everything for free
        return {
            "type": "object",
            "required": ["answer", "confidence"],
            "properties": {
                "answer":     {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }

    def generate(self, query: str, **kwargs) -> Any:
        # Call your LLM, DB, API, rule engine — anything
        return {"answer": "42", "confidence": 0.95}
```

Register and run:

```python
connector = AgentConnector(llm_client=llm)
connector.register(MyAgent())

result = connector.run("my_agent", "What is the meaning of life?")
print(result["output"])      # validated (and corrected if needed) output
print(result["corrected"])   # True if MASC had to fix something
print(result["anomaly"])     # anomaly dict, or None
```

---

## 🤖 Wrap Any LLM as an Agent (no subclassing)

```python
from examples.agents import LLMBackedAgent
from evolution.llm_protocol import OllamaAdapter

llm = OllamaAdapter(model="llama3")

agent = LLMBackedAgent(
    agent_id="qa_agent",
    description="Answers questions in JSON",
    llm_client=llm,
    system_prompt="You are a helpful assistant. Always reply in JSON with an 'answer' key.",
    output_schema={
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    },
)

connector.register(agent)
result = connector.run("qa_agent", "What is 2 + 2?")
```

---

## 🛡️ MASC — How Validation Works

MASC automatically derives validation rules from your agent's `output_schema`:

| Rule | Checks |
|------|--------|
| `null_output` | Output is not None/empty |
| `type_mismatch` | Top-level type matches schema `type` |
| `required_fields` | All `required` fields present |
| `property_types` | Field values match declared types |
| `numeric_range` | Numbers within `minimum`/`maximum` |
| `string_pattern` | Strings match declared `pattern` regex |
| `enum_violation` | Enum fields only use declared values |
| `empty_array` | Arrays meet `minItems` constraint |
| `string_not_empty` | Plain-text output is not blank |

**No schema?** MASC still validates: output is non-null and (if string) non-empty.

**Add a custom rule:**

```python
from interceptor.masc_validator import MASCValidator, ValidationRule

class ProfanityFilter(ValidationRule):
    name = "profanity_filter"
    description = "Blocks outputs containing banned terms."

    def check(self, output, schema):
        if isinstance(output, str) and "badword" in output.lower():
            return {"type": "profanity_filter", "detail": "Banned term found",
                    "field": None, "value": output}

validator = MASCValidator()
validator.add_rule(ProfanityFilter())
```

---

## 🧬 SePO — How Prompt Evolution Works

After `anomaly_threshold` consecutive anomalies on the same agent:

1. **With LLM**: SePO sends the current system prompt + anomaly details to your LLM and asks it to rewrite the prompt to prevent recurrence.
2. **Without LLM**: SePO appends a targeted instruction patch (e.g. "CRITICAL: Always return a non-empty response.").
3. The new prompt is saved to `logs/evolution_history.jsonl` and applied to the agent via `on_evolution()`.

---

## 🌐 API Server

```bash
python examples/server_demo.py
# → http://localhost:8000/docs  (Swagger UI)
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Run a query through an agent |
| GET | `/agents` | List registered agents |
| GET | `/stats` | Global + per-agent stats |
| GET | `/stats/{agent_id}` | Single agent stats |
| GET | `/logs?n=50` | Recent run logs |
| GET | `/evolution` | SePO evolution history |
| GET | `/evolution/{agent_id}` | Agent-specific evolution |
| GET | `/rules` | Active MASC rule names |
| GET | `/health` | Liveness probe |

---

## 📊 Dashboard

```bash
streamlit run dashboard/app.py
```

Pages: Overview KPIs · Agent Drill-Down · Evolution History · Live Logs · MASC Rules

---

## 📁 Project Structure

```
masc-sepo/
├── connectors/
│   ├── base_agent.py          # Abstract agent interface
│   └── agent_connector.py     # Central orchestrator
├── interceptor/
│   ├── masc_validator.py      # Schema-driven validation rules
│   └── correction_agent.py    # Heuristic + LLM corrections
├── evolution/
│   ├── llm_protocol.py        # LLM adapters (7 providers)
│   ├── sepo_engine.py         # Prompt evolution engine
│   └── evolution_tracker.py   # History analytics
├── logs/
│   └── logger.py              # Structured JSONL logger + stats
├── api/
│   └── server.py              # FastAPI application
├── dashboard/
│   └── app.py                 # Streamlit dashboard
└── examples/
    ├── agents.py              # Finance, Health, Legal, Sentiment, …
    ├── quickstart.py          # No-API-key demo
    └── server_demo.py         # Pre-loaded API server
```

---

## 📦 Installation

```bash
# Core only
pip install masc-sepo

# With dashboard
pip install "masc-sepo[dashboard]"

# With a specific LLM
pip install "masc-sepo[ollama]"
pip install "masc-sepo[openai]"
pip install "masc-sepo[anthropic]"

# Everything
pip install "masc-sepo[all]"
```

---

## License

MIT © 2024 — PRs welcome!
