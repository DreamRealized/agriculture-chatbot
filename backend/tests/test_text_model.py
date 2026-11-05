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
        for prefix in ("", "河北种植畜牧虫害害虫防治："):
            with self.subTest(prefix=prefix), contextlib.ExitStack() as stack:
                calls = []
                async def fake_ollama(**kwargs):
                    calls.append(kwargs)
                    yield {"response": "I will remember purple.", "done": False}
                    yield {"done": True}
                manager = UserMemoryManager()
                for name, value in dict(ENABLE_RAG=False, ENABLE_MCP=False,
                                        user_memory_manager=manager,
                                        generate_with_ollama_stream=fake_ollama).items():
                    stack.enter_context(patch.object(main, name, value))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                client = stack.enter_context(TestClient(main.app))
                first = prefix + "My favorite color is purple. Please remember it."
                second = "What is my favorite color?"
                for prompt in (first, second):
                    response = client.post("/analyze", json={"prompt": prompt, "chat_id": "memory-test-1"})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('"type": "done"', response.text)
                self.assertNotIn("Conversation history:", calls[0]["prompt"])
                self.assertIn("helpful general-purpose assistant", calls[0]["prompt"])
                self.assertNotIn("农业专家小羲", calls[0]["prompt"])
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

    def test_optional_rag_context_and_mcp_routing(self):
        calls = []
        tool_calls = []
        async def fake_ollama(**kwargs):
            calls.append(kwargs)
            yield {"response": "answer", "done": False}
            yield {"done": True}
        async def fake_analysis(**kwargs):
            yield "loading: rag_searching \n\n"
        async def fake_retrieve(**kwargs):
            return [{"text": "reference-fact"},
                    {"image": "reference.png", "title": "reference image"}]
        async def fake_tool(query, context, rag_result):
            tool_calls.append((query, rag_result))
            yield "tool answer"
        with contextlib.ExitStack() as stack:
            for name, value in dict(ENABLE_MCP=False, ENABLE_RAG=True,
                                    user_memory_manager=UserMemoryManager(),
                                    generate_with_ollama_stream=fake_ollama,
                                    run_rag_analyzing=fake_analysis,
                                    retrieveRAGResult=fake_retrieve,
                                    run_postgres_mcp_tool=fake_tool).items():
                stack.enter_context(patch.object(main, name, value, create=True))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            client = stack.enter_context(TestClient(main.app))
            response = client.post("/analyze", json={"prompt": "Summarize the reference", "chat_id": "rag"})
            self.assertIn('"type": "done"', response.text)
            self.assertIn("reference-fact", calls[-1]["prompt"])
            self.assertNotIn("reference.png", calls[-1]["prompt"])
            self.assertIn("reference.png", response.text)
            self.assertNotIn("小羲", response.text)
            with patch.object(main, "ENABLE_MCP", True), patch.object(main, "should_use_mcp_plugin", return_value=True):
                response = client.post("/analyze", json={"prompt": "database query", "chat_id": "mcp"})
            self.assertIn("tool answer", response.text)
            self.assertIn('"type": "done"', response.text)
            self.assertEqual(len(calls), 1)
            self.assertEqual(tool_calls, [("database query", [{"text": "reference-fact"}])])
            memory = main.user_memory_manager.get_memory("mcp", main.OLLAMA_TEXT_MODEL)
            self.assertEqual(memory.history[0]["output"], "tool answer")

    def test_image_conversation_history_and_followups(self):
        calls = []
        async def fake_ollama(**kwargs):
            calls.append(kwargs)
            yield {"response": "The chart shows purple bars.", "done": False}
            yield {"done": True}
        manager = UserMemoryManager()
        with contextlib.ExitStack() as stack:
            for name, value in dict(ENABLE_MCP=False, ENABLE_RAG=False,
                                    user_memory_manager=manager,
                                    generate_with_ollama_stream=fake_ollama).items():
                stack.enter_context(patch.object(main, name, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            client = stack.enter_context(TestClient(main.app))
            def send(prompt, images=None, chat_id="image-chat"):
                response = client.post("/analyze", json={"prompt": prompt, "images": images or [], "chat_id": chat_id})
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                self.assertIn('"type": "delta"', response.text)
                self.assertIn('"type": "done"', response.text)
                return calls[-1]
            send("Please answer briefly.")
            image_call = send("Explain this chart.", ["unique-base64-image-bytes"])
            self.assertEqual(image_call["model"], "qwen2.5vl:7b")
            self.assertEqual(image_call["image"], ["unique-base64-image-bytes"])
            self.assertEqual(image_call["prompt"].count("Please answer briefly."), 1)
            self.assertEqual(image_call["prompt"].count("Conversation history:"), 1)
            self.assertIn("documents, screenshots, charts", image_call["prompt"])
            self.assertIn("Do not invent details", image_call["prompt"])
            self.assertNotIn("农业专家", image_call["prompt"])
            followup = send("What color were the bars?")
            self.assertEqual(followup["model"], main.OLLAMA_TEXT_MODEL)
            self.assertEqual(followup["image"], [])
            self.assertEqual(followup["prompt"].count("Explain this chart."), 1)
            self.assertIn("The chart shows purple bars.", followup["prompt"])
            memory = manager.get_memory("image-chat", main.OLLAMA_TEXT_MODEL)
            self.assertNotIn("unique-base64-image-bytes", str(memory.history))
            isolated = send("What is shown?", ["other-image"], "separate-chat")
            self.assertNotIn("Explain this chart.", isolated["prompt"])
            self.assertNotIn("purple bars", isolated["prompt"])
            self.assertEqual(client.get("/clean_context/image-chat").status_code, 200)
            self.assertEqual(memory.history, [])
            cleared = send("Explain this new image.", ["new-image"])
            self.assertNotIn("Conversation history:", cleared["prompt"])
            self.assertNotIn("purple bars", cleared["prompt"])

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
