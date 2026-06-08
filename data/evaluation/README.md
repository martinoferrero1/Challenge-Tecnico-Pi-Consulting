# API run datasets

These JSONL datasets run ordered question sequences against the API using
content from `data/original_document.docx`.

The script prints and saves every response, and also calculates lightweight RAG
metrics for each run.

## Recommended mode/dataset pairs

### Direct document context

Use:

```env
CONVERSATION_CONTEXT_MODE=disabled
ANSWER_CACHE_MODE=document_context
```

Dataset:

```text
data/evaluation/direct_document_context.jsonl
```

This runs direct standalone questions where retrieval should be enough.

### Exact question cache

Use:

```env
CONVERSATION_CONTEXT_MODE=disabled
ANSWER_CACHE_MODE=question
```

Dataset:

```text
data/evaluation/question_cache_exact.jsonl
```

This repeats exact questions so you can observe cache behavior and latency.

### Conversational context-aware cache

Use:

```env
CONVERSATION_CONTEXT_MODE=rewrite
ANSWER_CACHE_MODE=context_aware
CONVERSATION_HISTORY_LIMIT=10
```

Dataset:

```text
data/evaluation/conversational_context_aware.jsonl
```

This sends multi-turn conversations without passing conversation history in
the request. The API keeps the conversation in memory while the process is
running.

## Running

Start the API with the `.env` configuration you want to test, then run:

```powershell
.\.venv\Scripts\python app\scripts\run_api_evaluation.py --dataset data\evaluation\conversational_context_aware.jsonl
```

Or run all datasets:

```powershell
.\.venv\Scripts\python app\scripts\run_api_evaluation.py
```

The script prints results in request order and writes JSONL output to:

```text
data/evaluation/results/
```

It also writes metrics summaries and Prometheus text files to:

```text
data/evaluation/metrics/
```

To group metrics from a run in a named folder:

```powershell
.\.venv\Scripts\python app\scripts\run_api_evaluation.py --dataset data\evaluation\question_cache_exact.jsonl --metrics-run-name cache-question-mode
```

That writes metrics to:

```text
data/evaluation/metrics/cache-question-mode/
```

Each run uses unique `user_name` values so in-memory conversations from one
run do not mix with another run.

## Metrics

The datasets include `expected_sections`, which makes these retrieval metrics
available:

- `Recall@K`
- `MRR`
- `Context relevance`

The script also computes automatic lexical approximations for:

- `Groundedness`
- `Answer relevance`

Operational metrics are measured directly:

- `Latency total`
- `Latency by stage`, when the API returns diagnostics
- `Estimated tokens per request`
- `Error rate`
- `Prompt injection attempts detected`
- `Cache hit rate`

`Citation accuracy` is emitted only when answers include bracket citations that
can be compared with `expected_sections`. The current answer prompt does not ask
for citations, so this metric will usually be `null`.

Use `--k` to change the K used by `Recall@K` and `MRR`:

```powershell
.\.venv\Scripts\python app\scripts\run_api_evaluation.py --dataset data\evaluation\direct_document_context.jsonl --k 4
```

The `.prom` files use Prometheus exposition format. To view them in Grafana, the
simplest path is to have Prometheus scrape or load those metrics, then add
Prometheus as a Grafana data source.
