# Local no-paid-API run

This forked setup avoids paid API keys by default:

- LLM calls use an OpenAI-compatible local endpoint, defaulting to Ollama at `http://localhost:11434/v1`.
- The original Serper/Google agent uses `SERPER_API_KEY` only when set. Without it, it falls back to DuckDuckGo web-search evidence.
- Wikipedia and news keyword extraction use local KeyBERT embeddings instead of OpenAI.
- The news agent uses `NYTIMES_API_KEY` only when set. Without it, it falls back to Wikipedia evidence.

## 1. Start a local LLM

Ollama is installed locally at `/home/ccl/Pratik/llm-exp/ollama`. Add it to
your shell and keep downloaded models in the same workspace:

```bash
export PATH=/home/ccl/Pratik/llm-exp/ollama/bin:$PATH
export OLLAMA_MODELS=/home/ccl/Pratik/llm-exp/ollama/models
ollama serve
```

If `ollama serve` is already running as a service, leave it running and continue.
In another terminal, use the same `PATH` and `OLLAMA_MODELS` exports, then pull a
model:

```bash
ollama pull llama3.1:8b
```

## 2. Create the Python environment

```bash
cd /home/ccl/Pratik/llm-exp/multi-agent-fact-checking
conda create -n mafc python=3.10 -y
conda activate mafc
pip install -r requirements.txt
```

The first run may download local embedding models from Hugging Face, such as `all-MiniLM-L6-v2`.

## 3. Configure the local model

The defaults are usually enough:

```bash
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export LOCAL_LLM_API_KEY=ollama
export LOCAL_LLM_MODEL=llama3.1:8b
```

You can use a different local OpenAI-compatible server by changing those variables. `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` are also supported and take precedence.

## 4. Run the API

```bash
python main.py
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/multi_agent/only_language_model \
  -H "Content-Type: application/json" \
  -d '{"raw":"Trump won the 2016 United States presidential election."}'
```

Then try the multi-agent route:

```bash
curl -X POST http://127.0.0.1:8000/multi_agent/fact_checking_binary \
  -H "Content-Type: application/json" \
  -d '{"raw":"Trump won the 2016 United States presidential election."}'
```

## Notes

- The included batch script `multi_agent_fact_checking_test.py` still expects datasets under `test_datasets/`, which are not present in this repository.
- Local LLMs may be less consistent than hosted models about returning exact Python dictionary text. If a request fails while parsing model output, retry with a stronger instruction-following model.
