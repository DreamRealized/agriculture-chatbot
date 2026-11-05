# General Chatbot MVP — Backend

FastAPI backend for general-purpose text chat, streaming responses, conversation
memory, and existing image support through Ollama. RAG retrieval and MCP database tools are
optional and **disabled by default**.

## Local setup

Use Python 3.10+ (the existing local environment uses Python 3.11).

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Start Ollama locally and download the models used by the existing chat paths:

```bash
ollama serve  # Skip if Ollama is already running.
# In another terminal:
ollama pull qwen3:4b-instruct
ollama pull qwen2.5vl:7b  # Required for image input.
```

Start the backend from `backend/`:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The server can start without RAG, MCP, or an active Ollama connection. Sending a
chat request requires Ollama and the selected model. Text requests use
`OLLAMA_TEXT_MODEL` (default `qwen3:4b-instruct`); image requests still use
`qwen2.5vl:7b`. Streaming, prompts, memory,
and context clearing retain their existing behavior.

## Configuration

No environment file is required. Defaults connect to Ollama at
`http://127.0.0.1:11434` and disable both optional integrations.

Configuration precedence, highest first:

1. Existing process/shell environment variables.
2. `backend/.env.<APP_ENV>` (defaults to `.env.dev`).
3. `backend/.env`.
4. Built-in defaults.

Set `APP_ENV` in the shell to select an environment-specific file. Files are
resolved relative to the backend directory. Keep local configuration and secrets
out of Git; `.env.example` contains only safe local defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVER_INFO` | `{"host":"127.0.0.1","port":11434}` | Ollama host and port as a JSON object. Partial objects inherit defaults. |
| `OLLAMA_TEXT_MODEL` | `qwen3:4b-instruct` | Text model for chat, RAG analysis, and MCP generation. Must be non-empty. |
| `ENABLE_RAG` | `false` | Enable the existing RAG analysis and retrieval path. |
| `ENABLE_MCP` | `false` | Enable MCP connections at startup and tool routing for matching queries. |

Flags accept `true/false`, `1/0`, `yes/no`, or `on/off` (case insensitive).
Invalid JSON, host/port values, or flags produce a configuration error without
printing the raw configuration.

To use a different text model, set `OLLAMA_TEXT_MODEL` in `.env` or the shell,
pull that exact model with `ollama pull <model>`, and restart the backend. The
same setting controls text conversation memory and context clearing; image
generation continues to use `qwen2.5vl:7b`.

To enable an integration, configure its existing settings in
`backend/config.jsonc`, start its service, and set the corresponding flag in
`.env` or the shell. Restart the backend after changing configuration.

- RAG: set `rag_url` and the relevant `rag_classification` entries, then set
  `ENABLE_RAG=true`. Image requests continue to skip RAG.
- MCP: configure `mcp_db_dict` URLs, keywords, and schemas, then set
  `ENABLE_MCP=true`. Enabled MCP services must be reachable during startup.

When disabled, the integration modules are not imported and their services are
not contacted by startup or chat requests. Each flag operates independently.

## API and components

- `POST /analyze`: accepts `prompt`, optional `images`, and `chat_id`; returns
  server-sent events.
- `GET /clean_context/{chat_id}`: clears that chat's model context.
- `utils/models.py`: Ollama generation helpers.
- `utils/promptsArchive.py`: existing prompt templates.
- `rag/rag.py`: optional retrieval integration.
- `mcp_plugins/`: optional MCP client and database tools.

## General text conversations

Every ordinary text request uses a neutral general-purpose prompt, including
messages containing former agriculture trigger keywords. Prior conversation
turns are included exactly once; reuse the same `chat_id` across turns. Memory
is held in process and resets when the backend restarts.

When RAG is explicitly enabled, retrieved text is supplied as reference material
for the assistant to use when relevant. Enabled MCP routes still take precedence
for matching database queries. Existing RAG topics, MCP database schemas and
keywords, and their specialized prompts remain unchanged. Legacy agricultural
templates remain available but are no longer selected by ordinary text or image chat.

Run backend regression tests from `backend/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## General image conversations

Attach images and ask about everyday objects, documents, screenshots, charts,
or other visual content. Image requests use a neutral prompt and the existing
`qwen2.5vl:7b` vision model; unclear visual details should be acknowledged.
Install the model locally with `ollama pull qwen2.5vl:7b`.

The frontend sends base64 images and uses one stable conversation ID for all
text/image requests and context clearing during the mounted chat session.
Image prompts include prior text exchanges exactly once. Image and text turns
share conversation history, but **image bytes are not stored in backend memory**.
Follow-ups can reference previous descriptions; reattach an image for renewed
visual analysis. Reloading the frontend starts a new conversation.

Image uploads and SSE streaming are unchanged. Image requests continue to bypass
optional RAG/MCP routing. Browser upload accepts `image/*`; actual format support
and readable detail depend on the vision model. Text-only requests continue to
use `OLLAMA_TEXT_MODEL` (default `qwen3:4b-instruct`).
