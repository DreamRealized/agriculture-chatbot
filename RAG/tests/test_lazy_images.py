import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models.models import QueryResponse

from RAG import retrieval


class LazyImageTests(unittest.TestCase):
    def setUp(self):
        # Import the actual API with inert dependencies; no network/model initialization.
        pipeline = ModuleType('pipeline')
        pipeline.COLLECTION_NAME = 'test-only'
        pipeline.client = Mock()
        pipeline.client.query_points.return_value = QueryResponse(points=[])
        pipeline.BgeEmbedder = Mock(return_value=Mock(embed=Mock(return_value=[0.0] * 1024)))
        self.vision = Mock()
        self.vision.embed.return_value = [0.0] * 512
        self.vision.embed_from_pil.return_value = [0.0] * 512
        pipeline.CLIPEmbedder = Mock(return_value=self.vision)
        self.pipeline = pipeline
        spec = importlib.util.spec_from_file_location(
            'rag_api_lazy_test', Path(__file__).resolve().parents[1] / 'main.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'pipeline': pipeline, 'retrieval': retrieval,
                                      spec.name: self.module}), patch('os.makedirs'):
            spec.loader.exec_module(self.module)
        self.addCleanup(self.module.get_image_embedder.cache_clear)

    @staticmethod
    def png():
        buffer = BytesIO()
        Image.new('RGB', (1, 1)).save(buffer, format='PNG')
        return buffer.getvalue()

    def test_startup_and_text_requests_never_initialize_clip(self):
        self.pipeline.CLIPEmbedder.assert_not_called()
        self.pipeline.client.query_points.return_value = QueryResponse(points=[
            ScoredPoint(id=1, version=1, score=0.8,
                        payload={'page_title': '资料', 'page_content': '正文'})])
        with patch.dict('os.environ', {}, clear=True), TestClient(self.module.app) as api:
            for _ in range(2):
                response = api.post('/search', json={'text': 'question'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()[0]['id'], '1')
                self.assertEqual(response.json()[0]['score'], 0.8)
        self.pipeline.CLIPEmbedder.assert_not_called()
        self.pipeline.BgeEmbedder.assert_called_once()

    def test_image_search_initializes_clip_once_then_reuses_it(self):
        import base64
        self.pipeline.CLIPEmbedder.assert_not_called()
        with TestClient(self.module.app) as api:
            for _ in range(2):
                response = api.post('/search', json={
                    'image_base64': base64.b64encode(self.png()).decode()})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), [])
        self.pipeline.CLIPEmbedder.assert_called_once()
        self.assertEqual(self.vision.embed_from_pil.call_count, 2)
        self.pipeline.client.upsert.assert_not_called()

    def test_embed_upload_initializes_clip_lazily_and_preserves_response(self):
        self.pipeline.CLIPEmbedder.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(self.module, 'UPLOAD_DIR', directory), TestClient(self.module.app) as api:
            response = api.post('/embed', data={'text': 'image caption'},
                                files={'image': ('test.png', self.png(), 'image/png')})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body['status'], 'success')
            self.assertTrue(Path(body['image_path']).is_file())
            self.vision.embed.assert_called_once_with(body['image_path'])
            point = self.pipeline.client.upsert.call_args.kwargs['points'][0]
            self.assertEqual(point.payload['text'], 'image caption')
            self.assertEqual(set(point.vector), {'text', 'image'})
            # A later image operation uses the same cached instance.
            self.assertIs(self.module.get_image_embedder(), self.vision)
        self.pipeline.CLIPEmbedder.assert_called_once()


if __name__ == '__main__':
    unittest.main()
