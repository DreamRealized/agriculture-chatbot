import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

from RAG.upload_structured_json import (
    chunk_id_for, chunk_text, document_id_for, ingest_documents, load_structured_json,
)


class FakeEmbedder:
    def __init__(self):
        self.inputs = []

    def embed(self, text):
        self.inputs.append(text)
        return [0.0] * 1024


class FakeClient:
    """An in-memory dictionary, with no Qdrant client or connection."""
    def __init__(self):
        self.points = {}
        self.writes = 0

    def upsert(self, *, collection_name, points, wait):
        assert collection_name == 'test-only'
        assert wait is True
        self.writes += 1
        self.points.update({point.id: point for point in points})


class IngestionTests(unittest.TestCase):
    def test_chunk_boundaries_overlap_and_full_coverage(self):
        text = '甲乙丙丁戊己庚辛壬癸子丑寅卯辰'
        chunks = chunk_text(text, chunk_size=6, chunk_overlap=2)
        self.assertEqual(chunks, [text[0:6], text[4:10], text[8:14], text[12:15]])
        self.assertEqual(chunks[0] + ''.join(chunk[2:] for chunk in chunks[1:]), text)
        self.assertTrue(all(len(chunk) <= 6 for chunk in chunks))
        self.assertEqual(chunk_text('abcdef', 6, 2), ['abcdef'])
        self.assertEqual(chunk_text('abc', 6, 2), ['abc'])
        self.assertEqual(chunk_text('abcdefg', 3, 0), ['abc', 'def', 'g'])
        self.assertEqual(chunk_text('  '), [])

    def test_invalid_chunk_settings_fail(self):
        for size, overlap in [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5),
                              (True, 0), (4, False), (4.5, 1)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_text('text', size, overlap)

    def test_ids_are_stable_valid_and_source_scoped(self):
        doc = document_id_for('source.json', '标题')
        self.assertEqual(doc, document_id_for('source.json', '标题'))
        self.assertNotEqual(doc, document_id_for('other.json', '标题'))
        self.assertNotEqual(doc, document_id_for('source.json', '其他标题'))
        point = chunk_id_for(doc, 0)
        self.assertEqual(point, chunk_id_for(doc, 0))
        self.assertNotEqual(point, chunk_id_for(doc, 1))
        self.assertEqual(uuid.UUID(point).version, 5)

    def test_loader_validates_records_and_duplicate_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.json'
            path.write_text(json.dumps([{'title': ' 标题 ', 'content': '正文'}]))
            self.assertEqual(load_structured_json(path), [('标题', '正文')])
            for data in [{}, [None], [{'title': 'x'}], [{'title': 'x', 'content': 12}],
                         [{'title': ' ', 'content': 'x'}],
                         [{'title': 'x', 'content': 'a'}, {'title': ' x ', 'content': 'b'}]]:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    load_structured_json(path)

    def test_repeatable_ingestion_and_metadata(self):
        docs = [('first', 'abcdefghijk'), ('second', 'short')]
        embedder, client = FakeEmbedder(), FakeClient()
        kwargs = dict(source_filename='seed.json', embedder=embedder, client=client,
                      collection_name='test-only', chunk_size=6, chunk_overlap=2)
        count = ingest_documents(docs, **kwargs)
        before = {key: point.model_dump() for key, point in client.points.items()}
        self.assertEqual(count, 4)
        self.assertEqual(ingest_documents(list(reversed(docs)), **kwargs), count)
        self.assertEqual(len(client.points), count)
        self.assertEqual(before, {key: point.model_dump() for key, point in client.points.items()})
        self.assertEqual(client.writes, 2)
        for point in client.points.values():
            payload = point.payload
            self.assertEqual(payload['source_filename'], 'seed.json')
            self.assertEqual(payload['document_id'], document_id_for('seed.json', payload['page_title']))
            self.assertEqual(point.id, chunk_id_for(payload['document_id'], payload['chunk_index']))
            self.assertLessEqual(len(payload['page_content']), 6)
            self.assertEqual(payload['chunk_size'], 6)
            self.assertEqual(payload['chunk_overlap'], 2)
            self.assertEqual(payload['chunk_count'], 3 if payload['page_title'] == 'first' else 1)
            self.assertEqual(set(point.vector), {'text'})
            self.assertEqual(len(point.vector['text']), 1024)

    def test_empty_or_failed_embedding_does_not_write(self):
        client = FakeClient()
        kwargs = dict(source_filename='seed.json', embedder=FakeEmbedder(),
                      client=client, collection_name='test-only')
        self.assertEqual(ingest_documents([], **kwargs), 0)
        class FailingEmbedder:
            def embed(self, text):
                raise RuntimeError('embedding failed')
        kwargs['embedder'] = FailingEmbedder()
        with self.assertRaisesRegex(RuntimeError, 'embedding failed'):
            ingest_documents([('title', 'content')], **kwargs)
        self.assertEqual(client.writes, 0)

    def test_actual_dataset_reimports_without_duplicates_using_fakes(self):
        path = Path(__file__).resolve().parents[1] / 'rag_text.json'
        docs = load_structured_json(path)
        self.assertEqual(len(docs), 56)
        embedder, client = FakeEmbedder(), FakeClient()
        kwargs = dict(source_filename=path.name, embedder=embedder, client=client,
                      collection_name='test-only')
        count = ingest_documents(docs, **kwargs)
        self.assertGreater(count, len(docs))
        self.assertEqual(ingest_documents(docs, **kwargs), count)
        self.assertEqual(len(client.points), count)
        self.assertTrue(all(len(text) <= 400 for text in embedder.inputs))
        self.assertEqual(len({p.payload['document_id'] for p in client.points.values()}), 56)

    def test_import_and_help_do_not_initialize_pipeline(self):
        root = Path(__file__).resolve().parents[2]
        code = ('import sys; import RAG.upload_structured_json; '
                'assert "pipeline" not in sys.modules; '
                'assert "RAG.pipeline" not in sys.modules; '
                'assert "sentence_transformers" not in sys.modules; '
                'assert "torch" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=root, check=True)
        result = subprocess.run([sys.executable, '-B', 'RAG/upload_structured_json.py', '--help'],
                                cwd=root, check=True, capture_output=True, text=True)
        self.assertIn('--chunk-size', result.stdout)
        self.assertIn('--chunk-overlap', result.stdout)


if __name__ == '__main__':
    unittest.main()
