# API run datasets

These JSONL datasets run ordered question sequences against the API using
content from `data/documento.docx`. They do not grade answer quality; they
only print and save API responses so you can inspect them manually.

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

Each run uses unique `user_name` values so in-memory conversations from one
run do not mix with another run.
