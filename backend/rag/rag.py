from env import OLLAMA_TEXT_MODEL
import httpx
from typing import Optional
from utils.promptsArchive import get_rag_analysis_prompt, NO_RELATION
from utils.models import generate_with_ollama
from utils.utils import clean_message
from utils.load_config import global_config

SUB_DOMAIN = "/search"

async def run_rag_analyzing(text: str, images: Optional[str] = None, top_k: int = 3):
    yield "loading: rag_analyzing \n\n"

    prompt = get_rag_analysis_prompt(text)
    llm_reply = await generate_with_ollama(
        model="qwen2.5vl:7b" if images else OLLAMA_TEXT_MODEL,
        prompt=prompt,
        image=images,
    )
    llm_reply = clean_message(llm_reply["response"])
    print("analyzing result: -------------------", llm_reply)

    if NO_RELATION in llm_reply:
        yield "loading: rag_no_relation \n\n"
    else:
        yield "loading: rag_searching \n\n"
    return



async def retrieveRAGResult(text: Optional[str] = None, image: Optional[str] = None, top_k: int = 3):
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            global_config.rag_url + SUB_DOMAIN,
            json={
                "text": text,
                "image_base64": image,
                "top_k": top_k
            },
        )
        return response.json()
