# Local Multi-Agent Fact Checking

This fork adapts `Ito-takayuki-lab/multi-agent-fact-checking` to run locally without paid API keys.

The local setup uses:

- Ollama through an OpenAI-compatible local endpoint
- DuckDuckGo web search as the no-key web-search fallback
- MediaWiki/Wikipedia retrieval for the Wikipedia agent
- KeyBERT for local keyword extraction
- Conda for the Python environment

## Quick Start

Start Ollama:

```bash
./scripts/start_ollama.sh
```

In another terminal, start the API:

```bash
./scripts/start_api.sh
```

Test the API:

```bash
curl -X POST http://127.0.0.1:8000/multi_agent/fact_checking_binary \
  -H "Content-Type: application/json" \
  -d '{"raw":"Trump won the 2016 United States presidential election."}'
```

## Documentation

- Local setup guide: [README_LOCAL.md](README_LOCAL.md)
- Migration report: [docs/local_no_paid_api_migration_report.md](docs/local_no_paid_api_migration_report.md)
