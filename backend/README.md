# 🌾 Agriculture Assistant — Backend

FastAPI backend for streaming text and image chat with Ollama, agricultural
prompts, and conversation memory. RAG retrieval and MCP database tools are
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
