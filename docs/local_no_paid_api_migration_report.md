# Local no-paid-API migration report

Date: 2026-05-31

This report summarizes the changes made to run `Ito-takayuki-lab/multi-agent-fact-checking` locally on this machine without paid API keys. The final setup uses a local Ollama-compatible LLM endpoint, DuckDuckGo web search for Agent 1, Wikipedia/MediaWiki for Agent 2, and a local LLM-only Agent 3.

## Starting point

The upstream repository was a research prototype with these practical blockers:

- OpenAI API key was hard-coded as an empty string.
- Serper/Google Search API key was hard-coded as an empty string.
- NYTimes API usage was embedded in the news agent.
- Several routes still returned experiment placeholder text instead of factuality labels.
- Some batch-test paths referenced removed datasets under `test_datasets/`.
- The pinned dependencies conflicted with modern `openai` requirements.
- The multi-agent route attempted to write experiment logs under a missing `experiment_results/SciFact/` directory.

The target constraint was: no paid API keys.

## Step 1: Local LLM support

File changed:

- `dependencies/language_model.py`

What changed:

- Replaced implicit OpenAI-cloud behavior with an OpenAI-compatible client configured by environment variables.
- Defaulted the base URL to Ollama: `http://localhost:11434/v1`.
- Defaulted the model to `llama3.1:8b`.
- Added support for:
  - `LOCAL_LLM_BASE_URL`
  - `LOCAL_LLM_API_KEY`
  - `LOCAL_LLM_MODEL`
  - `OPENAI_BASE_URL`
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
- Added a fallback tokenizer encoding for local model names not recognized by `tiktoken`.

Reason:

The original code required a hosted OpenAI API key. The local endpoint preserves the project’s existing OpenAI SDK call shape while allowing Ollama or another local OpenAI-compatible server.

## Step 2: Ollama local installation

Install location:

- `/home/ccl/Pratik/llm-exp/ollama`

What changed outside the repo:

- Installed the official Ollama Linux AMD64 package locally in the workspace.
- Created a local model directory at `/home/ccl/Pratik/llm-exp/ollama/models`.

Manual usage:

```bash
export PATH=/home/ccl/Pratik/llm-exp/ollama/bin:$PATH
export OLLAMA_MODELS=/home/ccl/Pratik/llm-exp/ollama/models
ollama serve
```

In another terminal:

```bash
export PATH=/home/ccl/Pratik/llm-exp/ollama/bin:$PATH
export OLLAMA_MODELS=/home/ccl/Pratik/llm-exp/ollama/models
ollama pull llama3.1:8b
```

Reason:

The user wanted a local install rather than system-wide installation.

## Step 3: Local run documentation

File added:

- `README_LOCAL.md`

What changed:

- Added local no-paid-API run instructions.
- Documented Ollama paths and environment variables.
- Documented Conda setup:

```bash
cd /home/ccl/Pratik/llm-exp/multi-agent-fact-checking
conda create -n mafc python=3.10 -y
conda activate mafc
pip install -r requirements.txt
```

Note:

We briefly tested Python `venv`, but the system Python lacked `ensurepip` because `python3.10-venv` was not installed. The user preferred reverting to Conda, so the docs now use Conda.

## Step 4: Dependency conflict fix

File changed:

- `requirements.txt`

What changed:

- Replaced:

```txt
typing_extensions==4.8.0
```

with:

```txt
typing_extensions>=4.11,<5
```

Reason:

`openai~=1.60.2` requires `typing-extensions>=4.11,<5`, but the upstream repo pinned `typing_extensions==4.8.0`, causing `pip install -r requirements.txt` to fail with `ResolutionImpossible`.

## Step 5: Wikipedia and news agents no longer use OpenAI keyword extraction

Files changed:

- `fact_check_agents/wikipedia_fact_check_agent.py`
- `fact_check_agents/news_fact_check_agent.py`

What changed:

- Replaced `KeyLLM` plus OpenAI-backed keyword extraction with local `KeyBERT`.
- Reused the existing `SentenceTransformer('all-MiniLM-L6-v2')` embedding model.
- Removed hard-coded NYTimes API key behavior.
- News agent now uses `NYTIMES_API_KEY` only if explicitly set.
- Without `NYTIMES_API_KEY`, the news agent falls back to Wikipedia-derived evidence.

Reason:

The original keyword extraction path still depended on hosted OpenAI calls. Local KeyBERT preserves keyword extraction without paid API usage.

## Step 6: Initial Serper-free fallback for Agent 1

File changed:

- `fact_check_agents/google_search_fact_check_agent.py`

What changed initially:

- Made `SERPER_API_KEY` optional via environment variable.
- If unset, Agent 1 originally fell back to Wikipedia evidence.

Reason:

The original Google search agent required a Serper API key. A fallback was needed to make multi-agent mode work without paid search APIs.

## Step 7: Replaced Agent 1 fallback with DuckDuckGo

File changed:

- `fact_check_agents/google_search_fact_check_agent.py`

What changed:

- Replaced the temporary Wikipedia fallback with DuckDuckGo HTML search.
- Uses `requests` and `BeautifulSoup`, both already present in the requirements.
- Parses DuckDuckGo results into the same Serper-like shape expected by the existing parser:

```python
{"organic": [{"title": ..., "snippet": ..., "link": ...}]}
```

Reason:

The temporary fallback made Agent 1 and Agent 2 too similar. DuckDuckGo restores a meaningful distinction:

- Agent 1: web-search evidence
- Agent 2: Wikipedia-specific evidence
- Agent 3: local LLM-generated evidence

Validation:

Using the `mafc` Conda environment, the DuckDuckGo fallback returned 5 results for:

```txt
result of the 2016 United States presidential election
```

including Wikipedia and NYTimes result snippets.

## Step 8: Robust MediaWiki helper

File added:

- `dependencies/wikipedia_client.py`

Files changed:

- `fact_check_agents/wikipedia_fact_check_agent.py`
- `fact_check_agents/news_fact_check_agent.py`

What changed:

- Added a small direct MediaWiki API client:
  - `search_titles()`
  - `page_summary()`
  - `page_url()`
- Handles HTTP and JSON parsing errors gracefully.

Reason:

The old `wikipedia` Python package crashed with:

```txt
requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

The direct MediaWiki helper avoids that brittle package behavior and returns empty results instead of crashing the route.

## Step 9: Fixed LLM agent claim shape

File changed:

- `fact_check_agents/language_model_agent.py`

What changed:

- Changed generated evidence records from storing the full claim dictionary:

```python
"claim": claim
```

to storing the claim text:

```python
"claim": claim["claim"]
```

Reason:

Later scoring and verification code expects claim text, not a nested dictionary.

## Step 10: Fixed route behavior and broken workflow calls

File changed:

- `routers/multiagent_factchecking.py`

What changed:

- Restored actual factuality-label responses in routes that still returned experiment placeholder text:

```txt
save comparison results
```

- Fixed `llm_agent_workflow()` calls that omitted the required `query_list`.
- Added query generation where needed before calling local LLM workflows.

Reason:

Several routes were still shaped for ablation experiments rather than API use. They either returned placeholders or called functions with missing arguments.

## Step 11: Experiment output directory creation

File changed:

- `dependencies/post_process.py`

What changed:

- Imported `os`.
- Added parent-directory creation before writing confidence-comparison logs:

```python
os.makedirs(os.path.dirname(confidence_comparison_save_dir), exist_ok=True)
```

Reason:

The multi-agent route completed all agents successfully but then crashed with:

```txt
FileNotFoundError: './experiment_results/SciFact/confidence_comparison_binary_save_dir.txt'
```

because the trimmed repo did not include `experiment_results/SciFact/`.

## Step 12: Runtime log labels clarified

File changed:

- `fact_check_agents/google_search_fact_check_agent.py`

What changed:

- Runtime labels now say `web_search` instead of `google`:
  - `google_evidence` -> `web_search_evidence`
  - `google_verification_result` -> `web_search_verification_result`
  - `google_search_fact_check_agent` -> `web_search_fact_check_agent`

Reason:

After switching the no-key fallback to DuckDuckGo, the old Google labels were misleading.

Note:

Internal class/file/function names were left alone to avoid broad import churn.

## Final agent architecture

The working no-paid-API multi-agent setup is:

```txt
Agent 1: DuckDuckGo web search evidence + local LLM judge
Agent 2: KeyBERT keyword extraction + MediaWiki/Wikipedia evidence + local LLM judge
Agent 3: local LLM-generated evidence + local LLM judge
```

The local LLM defaults to:

```txt
llama3.1:8b via http://localhost:11434/v1
```

## Working validation observed

The test claim:

```txt
Trump won the 2016 United States presidential election.
```

successfully ran through:

- Query generation
- Web-search evidence retrieval
- Wikipedia evidence retrieval
- LLM-only evidence generation
- Relevance scoring
- Per-agent factuality judgment
- Final MAFC score calculation

Observed score:

```txt
MAFC Score: 0.849975
```

## Runtime artifacts

The working run generated:

- `experiment_results/SciFact/confidence_comparison_binary_save_dir.txt`
- `data-gym-cache/`

These are runtime artifacts, not source changes.

## Changed source files

Tracked source files changed:

- `dependencies/language_model.py`
- `dependencies/post_process.py`
- `fact_check_agents/google_search_fact_check_agent.py`
- `fact_check_agents/language_model_agent.py`
- `fact_check_agents/news_fact_check_agent.py`
- `fact_check_agents/wikipedia_fact_check_agent.py`
- `requirements.txt`
- `routers/multiagent_factchecking.py`

New source/documentation files:

- `dependencies/wikipedia_client.py`
- `README_LOCAL.md`
- `docs/local_no_paid_api_migration_report.md`

## Current limitations

- The code still relies on local LLM output being parseable as Python dictionary text in several places.
- Agent 1 uses DuckDuckGo HTML scraping, which is free but may be less stable than an official API.
- Some internal names still reference Google for compatibility, even though runtime labels now say web search.
- Batch scripts still reference datasets not included in the trimmed upstream repository.
