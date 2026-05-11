# SHL Assessment Recommender

FastAPI service and evaluation toolkit for the SHL conversational assessment recommender assignment.

## What is included

- `GET /health` readiness endpoint
- `POST /chat` stateless chat endpoint with strict schema
- Local catalog ingestion and normalization from the provided SHL JSON feed
- Hybrid retrieval with keyword, BM25-like TF-IDF, and dense SVD similarity
- Guardrails for off-topic, legal/compliance, and prompt-injection requests
- Optional hosted LLM assist for state extraction and grounded reply writing
- Public trace replay harness and behavior probes
- Containerized deployment files
- PDF approach document generator

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/fetch_reference_data.py --skip-if-present
python -m uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API contract

### `GET /health`

Returns:

```json
{"status":"ok"}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer"}
  ]
}
```

Response:

```json
{
  "reply": "What level is the role and which skills matter most day one?",
  "recommendations": [],
  "end_of_conversation": false
}
```

## Hosted LLM mode

Hosted assistance is optional. The default hosted provider is Groq via its OpenAI-compatible `chat/completions` endpoint. If `LLM_API_KEY` is set, the app will use these defaults unless you override them:

- `LLM_BASE_URL=https://api.groq.com/openai/v1`
- `LLM_MODEL=llama-3.3-70b-versatile`

You can still override the provider by setting these variables explicitly:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

Without them, the service runs entirely in deterministic local fallback mode.

## Evaluation

Replay the public traces:

```bash
python scripts/replay_public_traces.py
```

Run behavior probes:

```bash
python scripts/run_behavior_probes.py
```

Generate the submission PDF:

```bash
python scripts/generate_approach_pdf.py
```
