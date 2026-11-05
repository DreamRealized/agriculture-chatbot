import json
import re

import httpx
from env import OLLAMA_TEXT_MODEL, SERVER_INFO


async def _raise_for_ollama_status(response):
    """Read streamed errors and report the response error, never the request body."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        await response.aread()
        try:
            body = response.json()
        except ValueError:
            detail = response.text.strip()
        else:
            detail = body.get("error", "") if isinstance(body, dict) else ""
        if not isinstance(detail, str) or not detail:
            detail = response.reason_phrase
        raise httpx.HTTPStatusError(
            f"Ollama HTTP {response.status_code}: {detail[:1000]}",
            request=response.request,
            response=response,
        ) from None


async def generate_with_ollama(
    prompt,
    model=OLLAMA_TEXT_MODEL,
    image=[],
    host=SERVER_INFO["host"],
    port=SERVER_INFO["port"],
):
    url = f"http://{host}:{port}/api/generate"
    payload = {"model": model, "prompt": prompt, "images": image, "stream": False, "temperature": 0.2}
    if image:
        payload["options"] = {"num_ctx": 8192}

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(url, json=payload)
            await _raise_for_ollama_status(response)
            return response.json()
    except httpx.HTTPStatusError as e:
        print(str(e))
        raise
    except httpx.RequestError as e:
        print(f"Network Error: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {str(e)}")
        raise
    except Exception as e:
        print(f"❗ Unexpected Error: {type(e).__name__}")
        raise


async def generate_with_ollama_stream(
    prompt,
    model=OLLAMA_TEXT_MODEL,
    image=[],
    host=SERVER_INFO["host"],
    port=SERVER_INFO["port"],
):
    url = f"http://{host}:{port}/api/generate"
    payload = {"model": model, "prompt": prompt, "images": image, "stream": True, "temperature": 0.2}
    if image:
        payload["options"] = {"num_ctx": 8192}

    async with httpx.AsyncClient(timeout=None, proxy=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            await _raise_for_ollama_status(response)
            async for line in response.aiter_lines():
                line = line.strip()
                print("line--------------", line)
                if line:
                    yield json.loads(line)


def clean_llm_response(text: str) -> str:
    """Remove <think>...</think> and strip whitespace."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
