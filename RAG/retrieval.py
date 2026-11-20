"""Text search and result formatting without model loading or database initialization."""
import math
import os
from pathlib import PurePosixPath

from qdrant_client.models import SearchParams

DEFAULT_TEXT_SCORE_THRESHOLD = 0.67


def text_score_threshold(value=None):
    """Read a process environment setting, or validate an explicit override."""
    if value is None:
        value = os.getenv('RAG_TEXT_SCORE_THRESHOLD', str(DEFAULT_TEXT_SCORE_THRESHOLD))
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError('RAG_TEXT_SCORE_THRESHOLD must be a finite number between 0 and 1') from None
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('RAG_TEXT_SCORE_THRESHOLD must be a finite number between 0 and 1')
    return threshold


def search_text(*, client, embedder, collection_name, text, top_k=3, score_threshold=None):
    """Use the existing BGE encoding and named text-vector search settings."""
    threshold = text_score_threshold(score_threshold)
    vector = embedder.embed(text)
    return client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=threshold,
        search_params=SearchParams(hnsw_ef=128),
        using='text',
    ).points


def format_search_results(points):
    """Keep raw payload fields and legacy aliases, adding canonical point ID/score.

    Preserve the existing content-based deduplication and incoming result order.
    Copy payloads so formatting never mutates Qdrant's returned records.
    """
    seen = set()
    results = []
    for point in points:
        payload = dict(point.payload or {})
        content_key = payload.get('text') or payload.get('page_content')
        if not content_key or content_key in seen:
            continue
        seen.add(content_key)
        payload['id'] = str(point.id)
        payload['score'] = float(point.score)
        if payload.get('image_path'):
            payload['image'] = PurePosixPath(payload['image_path']).name
            payload['title'] = payload.get('text', '')
        else:
            payload['title'] = payload.get('page_title', '')
            payload['content'] = payload.get('page_content', '')
        results.append(payload)
    return results
