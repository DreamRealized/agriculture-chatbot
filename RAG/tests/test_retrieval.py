import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Optional
from qdrant_client.models import ScoredPoint

from RAG.retrieval import format_search_results, search_text, text_score_threshold


def point(identifier, score, payload):
    return ScoredPoint(id=identifier, version=1, score=score, payload=payload)


class FakeEmbedder:
    def __init__(self):
        self.inputs = []

    def embed(self, text):
        self.inputs.append(text)
        return [0.0] * 1024


class FakeClient:
    """Apply the requested threshold/limit without a live Qdrant connection."""
    def __init__(self, points):
        self.points = points
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        matches = sorted((p for p in self.points if p.score >= kwargs['score_threshold']),
                         key=lambda p: p.score, reverse=True)
        return SimpleNamespace(points=matches[:kwargs['limit']])


class RetrievalTests(unittest.TestCase):
    def test_default_and_configurable_threshold(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(text_score_threshold(), 0.67)
        with patch.dict(os.environ, {'RAG_TEXT_SCORE_THRESHOLD': '0.70'}):
            self.assertEqual(text_score_threshold(), 0.70)
            self.assertEqual(text_score_threshold(0.65), 0.65)
        for value in ('nan', 'inf', '-0.1', '1.1', 'invalid', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'RAG_TEXT_SCORE_THRESHOLD'):
                text_score_threshold(value)

    def test_threshold_boundary_and_query_settings(self):
        client = FakeClient([point(1, 0.669, {}), point(2, 0.67, {}), point(3, 0.671, {})])
        embedder = FakeEmbedder()
        kwargs = dict(client=client, embedder=embedder, collection_name='multimodal', text='query')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual([p.id for p in search_text(**kwargs)], [3, 2])
        with patch.dict(os.environ, {'RAG_TEXT_SCORE_THRESHOLD': '0.65'}):
            self.assertEqual([p.id for p in search_text(**kwargs)], [3, 2, 1])
        call = client.calls[-1]
        self.assertEqual(call['using'], 'text')
        self.assertEqual(call['search_params'].hnsw_ef, 128)
        self.assertEqual(call['collection_name'], 'multimodal')
        self.assertEqual(call['limit'], 3)
        self.assertTrue(call['with_payload'])
        self.assertFalse(call['with_vectors'])
        self.assertEqual(call['query'], [0.0] * 1024)
        self.assertEqual(embedder.inputs, ['query', 'query'])

    def test_empty_results(self):
        client = FakeClient([point(1, 0.3, {'page_content': 'unrelated'})])
        results = search_text(client=client, embedder=FakeEmbedder(), collection_name='test',
                              text='question', score_threshold=0.7)
        self.assertEqual(format_search_results(results), [])
        self.assertEqual(format_search_results([]), [])

    def test_result_preserves_metadata_and_legacy_aliases_without_mutation(self):
        payload = {'page_title': '稻曲病', 'page_content': '参考内容', 'document_id': 'doc',
                   'source_filename': 'rag_text.json', 'chunk_index': 0, 'chunk_count': 2}
        original = copy.deepcopy(payload)
        record = point(12, 0.683, payload)
        result = format_search_results([record])[0]
        for key, value in original.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['id'], '12')
        self.assertEqual(result['score'], 0.683)
        self.assertEqual(result['title'], payload['page_title'])
        self.assertEqual(result['content'], payload['page_content'])
        self.assertEqual(record.payload, original)

    def test_image_aliases_and_content_deduplication_are_preserved(self):
        payload = {'text': '参考图片', 'image_path': './uploaded_images/example.png'}
        image = point(1, 0.8, payload)
        duplicate = point(1, 0.8, payload)
        text = point(2, 0.75, {'page_title': '资料', 'page_content': '正文'})
        results = format_search_results([image, duplicate, text])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['image'], 'example.png')
        self.assertEqual(results[0]['title'], '参考图片')
        self.assertEqual(results[0]['image_path'], payload['image_path'])
        self.assertEqual(results[0]['text'], payload['text'])
        self.assertNotIn('image', results[1])
        self.assertEqual(payload, image.payload)

    def test_image_threshold_stays_fixed_and_followup_text_uses_configuration(self):
        from io import BytesIO
        import base64
        from PIL import Image
        from qdrant_client.models import SearchParams

        image_point = point(1, 0.75, {'text': 'photo', 'image_path': './uploaded_images/photo.png'})
        text_point = point(2, 0.683, {'page_title': '资料', 'page_content': 'reference-fact'})
        class MultimodalClient(FakeClient):
            def query_points(self, **kwargs):
                if kwargs['using'] == 'image':
                    self.calls.append(kwargs)
                    return SimpleNamespace(points=[image_point])
                return super().query_points(**kwargs)
        client = MultimodalClient([text_point])
        embedder = FakeEmbedder()
        source = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
        nodes = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef))
                 and n.name in {'SearchRequest', 'unified_multimodal_search'}]
        app = FastAPI()
        namespace = dict(app=app, BaseModel=BaseModel, Optional=Optional, JSONResponse=JSONResponse,
                         client=client, text_embedder=embedder, COLLECTION_NAME='multimodal',
                         search_text=search_text, format_search_results=format_search_results,
                         Image=Image, BytesIO=BytesIO, base64=base64, SearchParams=SearchParams,
                         get_image_embedder=lambda: SimpleNamespace(embed_from_pil=lambda image: [0.0] * 512),
                         format_base64=lambda text: text, extract_scored_points=lambda response: response.points)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'RAG/main.py', 'exec'), namespace)
        buffer = BytesIO()
        Image.new('RGB', (1, 1)).save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        with patch.dict(os.environ, {'RAG_TEXT_SCORE_THRESHOLD': '0.67'}), TestClient(app) as api:
            response = api.post('/search', json={'image_base64': encoded})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([r['id'] for r in response.json()], ['1', '2'])
        self.assertEqual(client.calls[0]['using'], 'image')
        self.assertEqual(client.calls[0]['score_threshold'], 0.7)
        self.assertEqual(client.calls[1]['using'], 'text')
        self.assertEqual(client.calls[1]['score_threshold'], 0.67)
        self.assertEqual(embedder.inputs, ['photo的防治方法'])

    def test_search_route_remains_compatible_with_chatbot_consumers(self):
        # Execute the real route body without importing main's models/startup code.
        source = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
        nodes = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef))
                 and node.name in {'SearchRequest', 'unified_multimodal_search'}]
        client = FakeClient([
            point(1, 0.683, {'page_title': '稻曲病', 'page_content': 'reference-fact',
                            'document_id': 'doc', 'chunk_index': 0}),
            point(2, 0.75, {'text': 'photo', 'image_path': './uploaded_images/reference.png'}),
        ])
        app = FastAPI()
        namespace = dict(app=app, BaseModel=BaseModel, Optional=Optional, JSONResponse=JSONResponse,
                         client=client, text_embedder=FakeEmbedder(), COLLECTION_NAME='multimodal',
                         search_text=search_text, format_search_results=format_search_results)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'RAG/main.py', 'exec'), namespace)
        with patch.dict(os.environ, {'RAG_TEXT_SCORE_THRESHOLD': '0.67'}), TestClient(app) as api:
            response = api.post('/search', json={'text': '稻曲病', 'top_k': 3})
            self.assertEqual(response.status_code, 200)
            results = response.json()
            # The existing chatbot identifies images via .get('image') and reads aliases.
            images = [r for r in results if r.get('image')]
            references = [r for r in results if not r.get('image')]
            self.assertEqual(images[0]['image'], 'reference.png')
            self.assertEqual(images[0]['title'], 'photo')
            self.assertEqual(references[0]['title'], '稻曲病')
            self.assertEqual(references[0]['content'], 'reference-fact')
            self.assertEqual(references[0]['page_content'], 'reference-fact')
            self.assertEqual(references[0]['score'], 0.683)
            # Exercise the actual general prompt builder without unrelated config imports.
            prompt_source = ast.parse((Path(__file__).resolve().parents[2] / 'backend/utils/promptsArchive.py').read_text())
            prompt_node = next(n for n in prompt_source.body if isinstance(n, ast.FunctionDef)
                               and n.name == 'get_general_text_prompt')
            prompt_namespace = {}
            exec(compile(ast.Module(body=[prompt_node], type_ignores=[]), 'promptsArchive.py', 'exec'), prompt_namespace)
            prompt = prompt_namespace['get_general_text_prompt']('question', [], references)
            self.assertIn('reference-fact', prompt)
            # The calibrated default retains 0.683; an explicit 0.70 override rejects it.
            with patch.dict(os.environ, {}, clear=True):
                default_results = api.post('/search', json={'text': '稻曲病'}).json()
            self.assertEqual([r['id'] for r in default_results], ['2', '1'])
            with patch.dict(os.environ, {'RAG_TEXT_SCORE_THRESHOLD': '0.70'}):
                strict_results = api.post('/search', json={'text': '稻曲病'}).json()
            self.assertEqual([r['id'] for r in strict_results], ['2'])


if __name__ == '__main__':
    unittest.main()
