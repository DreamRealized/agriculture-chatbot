import argparse
import json
from pathlib import Path
import uuid

from qdrant_client.models import PointStruct

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80


def load_structured_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Expected a JSON list of title/content records')

    docs = []
    seen_titles = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ('title', 'content')
        ):
            raise ValueError(f'Record {index + 1} requires non-empty title and content strings')
        title = item['title'].strip()
        if title in seen_titles:
            raise ValueError(f'Duplicate title in source: {title}')
        seen_titles.add(title)
        docs.append((title, item['content']))
    return docs


def validate_chunk_settings(chunk_size, chunk_overlap):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if (not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool)
            or not 0 <= chunk_overlap < chunk_size):
        raise ValueError('chunk_overlap must be an integer from 0 to chunk_size - 1')


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split by characters, preserving exact overlap and all original content.

    Character counts are a simple bound, not an exact embedding-token budget.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def document_id_for(source_filename, title):
    # Do not use absolute paths or record positions: IDs survive moves and reordering.
    identity = json.dumps([source_filename, title], ensure_ascii=False)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'rag-document:' + identity))


def chunk_id_for(document_id, chunk_index):
    return str(uuid.uuid5(uuid.UUID(document_id), str(chunk_index)))


def ingest_documents(documents, *, source_filename, embedder, client, collection_name,
                     chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Upsert deterministic text points using injected embedding/database dependencies.

    Identical imports overwrite existing points. Changed documents/chunk settings
    can leave old trailing chunks; removal/reconciliation is a separate operation.
    Use distinct source filenames to distinguish separate knowledge sources.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)
    points = []
    for title, text in documents:
        document_id = document_id_for(source_filename, title)
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for index, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=chunk_id_for(document_id, index),
                payload={
                    # Retain keys consumed by the existing /search response formatter.
                    'page_title': title,
                    'page_content': chunk,
                    'document_id': document_id,
                    'source_filename': source_filename,
                    'chunk_index': index,
                    'chunk_count': len(chunks),
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                },
                vector={'text': embedder.embed(chunk)},
            ))

    # Build/validate every point before writing; embedding failure writes nothing.
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(points)


def main():
    parser = argparse.ArgumentParser(description='Import structured JSON as repeatable text chunks')
    parser.add_argument('filepath', nargs='?', type=Path,
                        default=Path(__file__).resolve().with_name('rag_text.json'))
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Maximum characters per chunk (default: 400)')
    parser.add_argument('--chunk-overlap', type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help='Overlapping characters (default: 80)')
    args = parser.parse_args()
    validate_chunk_settings(args.chunk_size, args.chunk_overlap)
    docs = load_structured_json(args.filepath)

    # pipeline initializes Qdrant; import it only for an explicitly requested run.
    if __package__:
        from .pipeline import BgeEmbedder, client, COLLECTION_NAME
    else:
        from pipeline import BgeEmbedder, client, COLLECTION_NAME
    from tqdm import tqdm

    count = ingest_documents(
        tqdm(docs, desc='嵌入并构建向量'),
        source_filename=args.filepath.name,
        embedder=BgeEmbedder(), client=client, collection_name=COLLECTION_NAME,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )
    print(f'上传完成：{count} 个文本块 → Qdrant <{COLLECTION_NAME}>')


if __name__ == '__main__':
    main()
