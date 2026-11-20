import asyncio
import contextlib
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
import main
from rag import rag
from utils.user_memory import UserMemoryManager


class RagTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_decisions(self):
        for reply in ('yes', ' YES ', '<think>reason</think>yes', 'no', '__NO_RELATION__', 'yes.', 'maybe', ''):
            with self.subTest(reply=reply), patch.object(rag, 'generate_with_ollama', AsyncMock(return_value={'response': reply})):
                events = [e async for e in rag.run_rag_analyzing('question')]
                expected = 'rag_searching' if reply in ('yes', ' YES ', '<think>reason</think>yes') else ('rag_no_relation' if reply in ('no', '__NO_RELATION__') else 'rag_unknown')
                self.assertIn(expected, events[-1])

    async def test_malformed_classifier_responses(self):
        for reply in (None, [], {}, {'response': None}, {'response': 1}):
            with self.subTest(reply=reply), patch.object(
                rag, 'generate_with_ollama', AsyncMock(return_value=reply)
            ), self.assertRaises(ValueError):
                [event async for event in rag.run_rag_analyzing('question')]

    async def test_classifier_timeout(self):
        async def slow(**kwargs):
            await asyncio.sleep(1)
        with patch.object(rag, 'generate_with_ollama', slow), patch.object(rag, 'RAG_CLASSIFICATION_TIMEOUT', 0.001):
            with self.assertRaises(TimeoutError):
                [e async for e in rag.run_rag_analyzing('question')]

    async def test_result_validation_and_bounded_http(self):
        real_client = httpx.AsyncClient
        valid = [{'title': 't', 'content': 'fact', 'id': 'id', 'score': 0.8}, {'title': 'i', 'image': 'a.png'}]
        for body in (valid, [], {}, None, ['bad'], [{}], [{'title': 't', 'content': 1}]):
            with self.subTest(body=body):
                def factory(**kwargs):
                    self.assertEqual(kwargs['timeout'].connect, 5)
                    self.assertEqual(kwargs['timeout'].read, 30)
                    return real_client(**kwargs, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
                with patch.object(rag.httpx, 'AsyncClient', factory):
                    if body == valid or body == []:
                        self.assertEqual(await rag.retrieveRAGResult(text='question'), body)
                    else:
                        with self.assertRaises(ValueError):
                            await rag.retrieveRAGResult(text='question')


class EndpointTests(unittest.TestCase):
    def send(self, enabled=True, images=None, decision='yes', failure=None, results=None, fallback=False):
        calls, requests = [], []
        real_client = httpx.AsyncClient
        classifier = AsyncMock(return_value={'response': decision})
        if failure in ('classification', 'classification_timeout'):
            classifier.side_effect = (TimeoutError('private details') if failure == 'classification_timeout'
                                      else RuntimeError('private details'))
        if failure == 'malformed_classifier':
            classifier.return_value = {'response': None}
        async def model(**kwargs):
            calls.append(kwargs)
            yield {'response': 'General answer.', 'done': False}
            yield {'done': True}
        def handler(request):
            requests.append(request)
            errors = {'connection': httpx.ConnectError, 'read_timeout': httpx.ReadTimeout, 'connect_timeout': httpx.ConnectTimeout}
            if failure in errors:
                raise errors[failure]('private details', request=request)
            if failure == '503':
                return httpx.Response(503, json={'error': 'private details'})
            if failure == 'json':
                return httpx.Response(200, content=b'not json')
            return httpx.Response(200, json=[] if results is None else results)
        def factory(**kwargs):
            return real_client(**kwargs, transport=httpx.MockTransport(handler))
        manager = UserMemoryManager()
        memory = manager.get_memory('test', main.OLLAMA_TEXT_MODEL)
        memory.save_context({'input': 'Earlier'}, {'response': 'Earlier answer'})
        original_history = list(memory.history)
        with contextlib.ExitStack() as stack:
            for name, value in dict(ENABLE_RAG=enabled, ENABLE_MCP=False, user_memory_manager=manager,
                                    run_rag_analyzing=rag.run_rag_analyzing, retrieveRAGResult=rag.retrieveRAGResult,
                                    generate_with_ollama_stream=model, log_async=lambda *a, **kw: None).items():
                stack.enter_context(patch.object(main, name, value, create=True))
            stack.enter_context(patch.object(rag, 'generate_with_ollama', classifier))
            stack.enter_context(patch.object(rag.httpx, 'AsyncClient', factory))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            client = stack.enter_context(TestClient(main.app))
            response = client.post('/analyze', json={'prompt': 'question', 'chat_id': 'test', 'images': images or []})
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/event-stream', response.headers['content-type'])
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        self.assertEqual(events[-1], {'type': 'done'})
        self.assertEqual(sum(e['type'] == 'done' for e in events), 1)
        self.assertFalse(any(e['type'] == 'error' for e in events))
        self.assertNotIn('private', response.text)
        if fallback:
            self.assertEqual(memory.history, original_history)
            self.assertEqual(calls, [])
            self.assertNotIn('General answer.', response.text)
            notices = [e['token'] for e in events if e['type'] == 'delta' and not e['token'].startswith('loading:')]
            self.assertEqual(len(notices), 1)
            self.assertTrue(notices[0].startswith("I couldn't"))
        else:
            self.assertEqual(memory.history[-1]['output'], 'General answer.')
        return (calls[0] if calls else None), requests, classifier, response.text

    def test_disabled_and_image_bypass(self):
        for kwargs in ({'enabled': False}, {'images': ['image-data']}):
            call, requests, classifier, _ = self.send(**kwargs)
            classifier.assert_not_called()
            self.assertEqual(requests, [])
            self.assertEqual(call['model'], 'qwen2.5vl:7b' if kwargs.get('images') else main.OLLAMA_TEXT_MODEL)

    def test_confirmed_general_questions(self):
        for decision in ('no', '__NO_RELATION__'):
            call, requests, _, _ = self.send(decision=decision)
            self.assertEqual(requests, [])
            self.assertNotIn('Retrieved reference material (', call['prompt'])

    def test_unknown_and_empty_results(self):
        for decision in ('yes', 'unexpected', 'yes.', ''):
            with self.subTest(decision=decision):
                _, requests, _, text = self.send(decision=decision, fallback=True)
                self.assertEqual(len(requests), int(decision == 'yes'))
                expected = 'retrieve usable agricultural references' if decision == 'yes' else 'determine whether'
                self.assertIn(expected, text)

    def test_grounding_metadata(self):
        call, requests, _, _ = self.send(results=[{'title': 'Title', 'content': 'verified fact', 'page_content': 'verified fact', 'page_title': 'Title', 'id': 'point-123', 'score': 0.8, 'source_filename': 'source.json'}])
        self.assertEqual(len(requests), 1)
        self.assertEqual(call['prompt'].count('verified fact'), 1)
        self.assertIn('Retrieved reference material (', call['prompt'])
        for value in ('verified fact', 'point-123', 'source.json'):
            self.assertIn(value, call['prompt'])

    def test_failures_fall_back_and_complete(self):
        for failure in ('classification', 'classification_timeout', 'malformed_classifier', '503', 'connection', 'read_timeout', 'connect_timeout', 'json', 'shape', 'item'):
            with self.subTest(failure=failure):
                call, _, _, text = self.send(fallback=True, failure=failure, results={'error': 'bad'} if failure == 'shape' else ['bad'])
                self.assertIn('rag_unavailable', text)
                self.assertIsNone(call)

    def test_fallback_is_deterministic(self):
        def notice(text):
            events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
            return [e['token'] for e in events if e['type'] == 'delta' and not e['token'].startswith('loading:')]
        empty = self.send(fallback=True)[-1]
        outage = self.send(fallback=True, failure='503')[-1]
        self.assertEqual(notice(empty), notice(outage))
        unknown = self.send(fallback=True, decision='maybe')[-1]
        failed = self.send(fallback=True, failure='classification')[-1]
        self.assertEqual(notice(unknown), notice(failed))
        self.assertNotEqual(notice(empty), notice(unknown))

    def test_image_only_references_do_not_enable_unsupported_text_answer(self):
        self.send(fallback=True, results=[{'title': 'Photo', 'image': 'photo.png'}])
