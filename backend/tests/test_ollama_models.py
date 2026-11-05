import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
from utils import models


class OllamaModelTests(unittest.IsolatedAsyncioTestCase):
    async def call_helper(self, streaming, images, handler):
        # Exercise the real HTTP request construction with an in-memory transport.
        factory = httpx.AsyncClient
        with patch.object(models.httpx, 'AsyncClient',
                          side_effect=lambda **kwargs: factory(
                              transport=httpx.MockTransport(handler), **kwargs)), \
                contextlib.redirect_stdout(io.StringIO()):
            kwargs = dict(prompt='private prompt', image=images,
                          model='qwen2.5vl:7b' if images else 'custom-text-model')
            if streaming:
                return [item async for item in models.generate_with_ollama_stream(**kwargs)]
            return await models.generate_with_ollama(**kwargs)

    async def test_vision_context_and_unchanged_text_payloads(self):
        for streaming in (False, True):
            for images in ([], ['private-image-bytes']):
                with self.subTest(streaming=streaming, images=bool(images)):
                    def handler(request):
                        payload = json.loads(request.content)
                        expected = dict(model='qwen2.5vl:7b' if images else 'custom-text-model',
                                        prompt='private prompt', images=images,
                                        stream=streaming, temperature=0.2)
                        if images:
                            expected['options'] = {'num_ctx': 8192}
                        self.assertEqual(payload, expected)
                        return httpx.Response(200, content=b'{"response":"hello","done":false}\n{"done":true}\n'
                                              if streaming else b'{"response":"hello"}')
                    result = await self.call_helper(streaming, images, handler)
                    self.assertEqual(result[0]['response'] if streaming else result['response'], 'hello')
                    if streaming:
                        self.assertTrue(result[-1]['done'])

    async def test_complete_useful_error_without_request_payload(self):
        detail = 'request (4154 tokens) exceeds the available context size (4096 tokens), try increasing it'
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                def handler(request):
                    return httpx.Response(400, json={'error': detail,
                                                    'prompt': 'private prompt',
                                                    'images': ['private-image-bytes']})
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    await self.call_helper(streaming, ['private-image-bytes'], handler)
                message = str(caught.exception)
                self.assertIn(detail, message)
                self.assertIn('400', message)
                self.assertNotIn('private prompt', message)
                self.assertNotIn('private-image-bytes', message)

    async def test_plain_text_error_and_missing_error_fallback(self):
        for body, expected in [(b'vision context unavailable', 'vision context unavailable'),
                               (b'{"prompt":"private prompt"}', 'Bad Request')]:
            with self.subTest(body=body):
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    await self.call_helper(True, [], lambda request: httpx.Response(400, content=body))
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn('private prompt', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
