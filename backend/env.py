import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Shell variables take precedence over environment-specific and shared files.
env = os.getenv("APP_ENV", "dev")
backend_dir = Path(__file__).resolve().parent
load_dotenv(backend_dir / f".env.{env}")
load_dotenv(backend_dir / ".env")

# Preserve the existing SERVER_INFO interface, including partial overrides.
SERVER_INFO = {"host": "127.0.0.1", "port": 11434}
SERVER_INFO_STR = os.getenv("SERVER_INFO", "").strip()
if SERVER_INFO_STR:
    try:
        overrides = json.loads(SERVER_INFO_STR)
    except json.JSONDecodeError:
        raise ValueError("SERVER_INFO must be a JSON object with host and/or port") from None
    if not isinstance(overrides, dict):
        raise ValueError("SERVER_INFO must be a JSON object with host and/or port")
    SERVER_INFO.update(overrides)

if not isinstance(SERVER_INFO["host"], str) or not SERVER_INFO["host"].strip():
    raise ValueError("SERVER_INFO host must be a non-empty hostname or IP address")
try:
    port = SERVER_INFO["port"]
    if isinstance(port, bool) or not isinstance(port, (int, str)):
        raise ValueError
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError
except (TypeError, ValueError):
    raise ValueError("SERVER_INFO port must be an integer between 1 and 65535") from None
SERVER_INFO["port"] = port


def _enabled(name):
    value = os.getenv(name, "false").strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"true", "1", "yes", "on"}


ENABLE_RAG = _enabled("ENABLE_RAG")
ENABLE_MCP = _enabled("ENABLE_MCP")

# Shared by text generation, optional integrations, and conversation memory keys.
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen3:4b-instruct").strip()
if not OLLAMA_TEXT_MODEL:
    raise ValueError("OLLAMA_TEXT_MODEL must be a non-empty model name")
