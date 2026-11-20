# Agricultural text RAG service

This optional FastAPI service retrieves agricultural reference chunks from Qdrant.
The general chatbot uses it only when `ENABLE_RAG=true`. Its current knowledge
base is `RAG/rag_text.json`: 56 documents, producing 100 chunks with the defaults.

## Python and dependencies

From the project root (Python 3.11 is the locally validated version):

```bash
cd RAG
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`qdrant-client==1.20.0` is validated with Qdrant server 1.20.2.
`python-multipart` is required to register the upload route even for text-only
startup. Most other dependencies are not pinned; this is not a full lockfile.

## Qdrant

Start Docker Desktop, then provision Qdrant if you do not already have an instance:

```bash
docker run -d --name agriculture-qdrant \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v agriculture-qdrant-data:/qdrant/storage qdrant/qdrant:v1.20.2
```

This downloads the Docker image if missing. Do not start a second instance over
an existing setup. HTTP/dashboard uses port 6333; the Python client prefers gRPC
on port 6334. `pipeline.py` connects to localhost and initializes the `multimodal`
collection if absent. Existing collections are reused, not schema-migrated.
The named vectors are `text` (1024 dimensions) and `image` (512), both cosine.

## Models and downloads

Text uses `BAAI/bge-large-zh-v1.5` through SentenceTransformer with normalized
embeddings. BGE loads at API startup or ingestion. The first uncached load needs
network access and a large download (roughly 1.3 GB for weights, plus supporting
files); allow additional disk space and RAM. It has been validated on the local
Apple Silicon machine. Cached models can be used offline.

CLIP (`openai/clip-vit-base-patch32`) is **optional for text-only operation**.
`main.py:get_image_embedder()` initializes it lazily and caches it per process.
Provision it separately before using image `/search` or `/embed`; those operations
can otherwise trigger a download, or fail if offline. No CLIP download is needed
for text startup, ingestion, or search.

## Text ingestion CLI

From `RAG`, inspect help without loading models or connecting to Qdrant:

```bash
.venv/bin/python upload_structured_json.py --help
```

Run the following only when intentionally importing data; it writes to Qdrant
and can download BGE if uncached:

```bash
.venv/bin/python upload_structured_json.py rag_text.json \
  --chunk-size 400 --chunk-overlap 80
```

The input is a JSON list of nonempty `title`/`content` strings with unique titles.
Chunks use character counts, not token counts. IDs are deterministic from source
filename, title, and chunk index, so identical repeated imports overwrite the same
points. Payloads retain title/content, document ID, source filename, chunk index,
count, size, and overlap. Changed titles, shorter documents, or changed chunk
settings can leave stale points; reconciliation is not implemented. Different
sources must use distinct filenames. Ingestion builds all vectors before one
upsert; it is intended for this small dataset, not unlimited bulk imports.

## Start the API on port 8100

From `RAG`, using cached BGE without unexpected downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8100
```

If BGE is not cached, offline startup fails; provision it intentionally first.

```bash
curl -sS http://127.0.0.1:8100/search \
  -H 'Content-Type: application/json' \
  -d '{"text":"稻曲病有哪些症状？","top_k":3}'
```

`/search` accepts `text`, optional `image_base64`, and `top_k` (default 3).
Results are a list; no matches returns `[]`. Each result preserves original
payload metadata and adds string `id`, numeric `score`, and legacy `title`/`content`
(or `title`/`image`) aliases. Scores are similarities, not correctness probabilities.
`retrieval.py` handles text search and response formatting; `pipeline.py` holds
embedding and Qdrant configuration. `/embed` accepts multipart text plus an image.
The optional uploader lives in `frontend/rag_uploader`; run `pnpm dev:rag` from
`frontend` to start it.

### Similarity threshold

Set `RAG_TEXT_SCORE_THRESHOLD` in the RAG service's process environment, for example
prefix the API command with `RAG_TEXT_SCORE_THRESHOLD=0.70`. It must be finite and
between 0 and 1. Default **0.67** retained expected documents for 13/14 relevant
queries in a small 31-query calibration, versus 6/14 at 0.70. This is a development
calibration, not an accuracy guarantee; `玉米螟` still falls below the threshold.
Search ranks chunks with content-based deduplication, not one result per document.
Image similarity remains 0.7; image-triggered follow-up text search uses the
configurable text threshold. The service does not load backend `.env` files.

## Enable chatbot integration

Keep Ollama on port 11434 with the existing `qwen3:4b-instruct` model installed.
In a separate terminal, from the project root:

```bash
cd backend
APP_ENV=dev ENABLE_RAG=true ENABLE_MCP=false \
  .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

This assumes the existing backend environment is installed; see `backend/README.md`.
The RAG URL is configured in `backend/config.jsonc` (`http://localhost:8100`).
RAG and MCP remain disabled by default. Image chat bypasses RAG and uses the
existing Ollama vision model independently of CLIP.

With RAG enabled, explicit `yes` classification retrieves agricultural references;
explicit `no` or `__NO_RELATION__` continues normal general chat. Unknown/malformed
classification or classification failure returns a deterministic retry notice.
Agricultural requests with empty, invalid, timed-out, or failed retrieval return a
deterministic knowledge-unavailable notice, without model generation or citations.
Only usable text references permit grounded Qwen generation. These operational
notices leave conversation memory unchanged. SSE retains `delta` events and exactly
one `done` on these paths. RAG-disabled text chat and image chat keep their existing
behavior. References retain source metadata in prompts without duplicate content
aliases; the public search response remains backward compatible.

## Tests and limitations

From the project root, without models or live database writes:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RAG/.venv/bin/python -m unittest discover -s RAG/tests -v
(cd backend && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v)
```

Classification is model-based and examines the current message, not a contextual
query rewrite. False classifications and missed relevant chunks remain possible.
Grounding improves evidence but does not guarantee factual answers. Unknown
requests conservatively receive a retry notice even if they were general questions.
Fallback notices are bilingual English/Chinese and are not persisted in memory.
Image retrieval retains legacy agricultural follow-up logic; real CLIP inference
has not been validated in this text-only phase. Model inference/database calls in
the RAG API are synchronous and can block concurrent requests.

These are local development services: authentication, input/upload limits, strict
CORS, sanitized server errors, and production logging controls are not implemented.
Do not expose them publicly; existing backend debug output includes prompts and
reference content. Qdrant data/model caches and local environment files are not
application source and should not be committed.
