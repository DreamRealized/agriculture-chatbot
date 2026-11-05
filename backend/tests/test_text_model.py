import contextlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main
from env import OLLAMA_TEXT_MODEL
from utils.models import generate_with_ollama, generate_with_ollama_stream
from utils.user_memory import UserMemoryManager


class TextModelTests(unittest.TestCase):
    def test_configuration(self):
        source = Path(__file__).resolve().parents[1] / "env.py"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "env.py"
            target.write_text(source.read_text())
            def load(values):
                with patch.dict(os.environ, values, clear=True):
                    spec = importlib.util.spec_from_file_location("test_env", target)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.OLLAMA_TEXT_MODEL
            self.assertEqual(load({}), "qwen3:4b-instruct")
            (target.parent / ".env").write_text("OLLAMA_TEXT_MODEL=file-model\n")
            self.assertEqual(load({}), "file-model")
            self.assertEqual(load({"OLLAMA_TEXT_MODEL": "custom-model"}), "custom-model")
            with self.assertRaisesRegex(ValueError, "OLLAMA_TEXT_MODEL"):
                load({"OLLAMA_TEXT_MODEL": "   "})

    def test_helper_defaults(self):
        for helper in (generate_with_ollama, generate_with_ollama_stream):
            self.assertEqual(inspect.signature(helper).parameters["model"].default, OLLAMA_TEXT_MODEL)

    def test_two_turn_history_reaches_ollama(self):
        for enhanced in (False, True):
            with self.subTest(enhanced=enhanced), contextlib.ExitStack() as stack:
                calls = []
                async def fake_ollama(**kwargs):
                    calls.append(kwargs)
                    yield {"response": "I will remember purple.", "done": False}
                    yield {"done": True}
                manager = UserMemoryManager()
                for name, value in dict(ENABLE_RAG=False, ENABLE_MCP=False,
                                        user_memory_manager=manager,
                                        should_apply_enhanced_prompt=lambda _: enhanced,
                                        generate_with_ollama_stream=fake_ollama).items():
                    stack.enter_context(patch.object(main, name, value))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                client = stack.enter_context(TestClient(main.app))
                first = "My favorite color is purple. Please remember it."
                second = "What is my favorite color?"
                for prompt in (first, second):
                    response = client.post("/analyze", json={"prompt": prompt, "chat_id": "memory-test-1"})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('"type": "done"', response.text)
                self.assertNotIn("Conversation history:", calls[0]["prompt"])
                self.assertEqual(calls[1]["prompt"].count(first), 1)
                self.assertEqual(calls[1]["prompt"].count("I will remember purple."), 1)
                self.assertIn(second, calls[1]["prompt"])
                memory = manager.get_memory("memory-test-1", main.OLLAMA_TEXT_MODEL)
                self.assertEqual([turn["input"] for turn in memory.history], [first, second])
                client.post("/analyze", json={"prompt": second, "chat_id": "different-chat"})
                self.assertNotIn("purple", calls[-1]["prompt"])
                client.get("/clean_context/memory-test-1")
                client.post("/analyze", json={"prompt": second, "chat_id": "memory-test-1"})
                self.assertNotIn("purple", calls[-1]["prompt"])

    def test_streaming_memory_and_images(self):
        calls = []
        async def fake_ollama(**kwargs):
            calls.append(kwargs)
            yield {"response": "hello", "done": False}
            yield {"done": True}
        manager = UserMemoryManager()
        with contextlib.ExitStack() as stack:
            for name, value in dict(OLLAMA_TEXT_MODEL="custom-text-model", ENABLE_RAG=False,
                                    ENABLE_MCP=False, user_memory_manager=manager,
                                    generate_with_ollama_stream=fake_ollama).items():
                stack.enter_context(patch.object(main, name, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            client = stack.enter_context(TestClient(main.app))
            for prompt in ("first", "second"):
                response = client.post("/analyze", json={"prompt": prompt, "chat_id": "test"})
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                self.assertIn('"token": "hello"', response.text)
                self.assertIn('"type": "done"', response.text)
                self.assertEqual(calls[-1]["model"], "custom-text-model")
            memory = manager.get_memory("test", "custom-text-model")
            self.assertEqual([entry["input"] for entry in memory.history], ["first", "second"])
            response = client.post("/analyze", json={"prompt": "image", "images": ["image-data"], "chat_id": "test"})
            self.assertIn('"type": "done"', response.text)
            self.assertEqual(calls[-1]["model"], "qwen2.5vl:7b")
            self.assertEqual(calls[-1]["image"], ["image-data"])
            self.assertEqual(client.get("/clean_context/test").status_code, 200)
            self.assertEqual(memory.history, [])


if __name__ == "__main__":
    unittest.main()
