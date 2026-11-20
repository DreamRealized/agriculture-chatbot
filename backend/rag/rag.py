import asyncio
from typing import Optional

import httpx
from env import OLLAMA_TEXT_MODEL
from utils.promptsArchive import get_rag_analysis_prompt
from utils.models import generate_with_ollama
from utils.utils import clean_message
from utils.load_config import global_config

SUB_DOMAIN = "/search"
RAG_CLASSIFICATION_TIMEOUT = 60.0
RAG_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


async def run_rag_analyzing(text: str, images: Optional[str] = None, top_k: int = 3):
    yield "loading: rag_analyzing \n\n"
    prompt = get_rag_analysis_prompt(text)
    llm_reply = await asyncio.wait_for(
        generate_with_ollama(
            model="qwen2.5vl:7b" if images else OLLAMA_TEXT_MODEL,
            prompt=prompt,
            image=images,
        ),
        timeout=RAG_CLASSIFICATION_TIMEOUT,
    )
    if not isinstance(llm_reply, dict) or not isinstance(llm_reply.get("response"), str):
        raise ValueError("Invalid RAG classification response")
    decision = clean_message(llm_reply["response"]).strip().casefold()
    # Accept only the explicit affirmative; no, legacy sentinels, and surprises skip.
    if decision == "yes":
        yield "loading: rag_searching \n\n"
    elif decision in ("no", "__no_relation__"):
        yield "loading: rag_no_relation \n\n"
    else:
        yield "loading: rag_unknown \n\n"


def validate_rag_results(results):
    """Validate consumer-required aliases while preserving optional source metadata."""
    if not isinstance(results, list):
        raise ValueError("RAG response must be a list of result objects")
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("RAG results must be objects")
        fields = ("title", "image") if "image" in result else ("title", "content")
        if not all(isinstance(result.get(field), str) and result[field].strip()
                   for field in fields):
            raise ValueError("RAG result is missing valid title/content or title/image fields")
    return results


async def retrieveRAGResult(text: Optional[str] = None, image: Optional[str] = None, top_k: int = 3):
    async with httpx.AsyncClient(timeout=RAG_HTTP_TIMEOUT) as client:
        response = await client.post(
            global_config.rag_url + SUB_DOMAIN,
            json={"text": text, "image_base64": image, "top_k": top_k},
        )
        response.raise_for_status()
        return validate_rag_results(response.json())
