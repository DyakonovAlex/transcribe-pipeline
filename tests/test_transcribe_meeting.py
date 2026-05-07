import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "transcribe_meeting.py"
SPEC = importlib.util.spec_from_file_location("transcribe_meeting", MODULE_PATH)
transcribe_meeting = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(transcribe_meeting)


class ValidateConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

        self.whisper_cli = self.tmp_path / "whisper-cli"
        self.whisper_cli.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        self.whisper_cli.chmod(0o755)

        self.model_path = self.tmp_path / "model.bin"
        self.model_path.write_text("model", encoding="utf-8")

        self.notes_dir = self.tmp_path / "notes"
        self.notes_dir.mkdir()

        self.prompt_template = self.tmp_path / "prompt.txt"
        self.prompt_template.write_text("{{transcription}}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _base_config(self):
        return {
            "whisper_cli": str(self.whisper_cli),
            "model_path": str(self.model_path),
            "whisper_args": {
                "language": "ru",
                "threads": 2,
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "max_len": 1,
                "no_timestamps": True,
            },
            "output_dir": str(self.tmp_path / "out"),
            "notes_dir": str(self.notes_dir),
            "placeholder": "{{input}}",
            "ollama": {
                "url": "http://localhost:11434/api/generate",
                "model": "test-model",
                "prompt_template": str(self.prompt_template),
            },
        }

    def test_validate_config_ok_without_ollama(self):
        config = self._base_config()
        transcribe_meeting.validate_config(config, require_llm=False)

    def test_validate_config_missing_root_key(self):
        config = self._base_config()
        del config["model_path"]

        with self.assertRaises(SystemExit) as ctx:
            transcribe_meeting.validate_config(config, require_llm=False)

        self.assertIn("model_path", str(ctx.exception))

    def test_validate_config_missing_whisper_arg(self):
        config = self._base_config()
        del config["whisper_args"]["beam_size"]

        with self.assertRaises(SystemExit) as ctx:
            transcribe_meeting.validate_config(config, require_llm=False)

        self.assertIn("whisper_args.beam_size", str(ctx.exception))

    def test_validate_config_checks_paths(self):
        config = self._base_config()
        config["whisper_cli"] = str(self.tmp_path / "missing-whisper-cli")

        with self.assertRaises(SystemExit) as ctx:
            transcribe_meeting.validate_config(config, require_llm=False)

        self.assertIn("whisper_cli", str(ctx.exception))

    def test_validate_config_requires_ollama_keys(self):
        config = self._base_config()
        del config["ollama"]["prompt_template"]

        with self.assertRaises(SystemExit) as ctx:
            transcribe_meeting.validate_config(config, require_llm=True, llm_provider="ollama")

        self.assertIn("ollama.prompt_template", str(ctx.exception))

    def test_validate_config_requires_gemini_cli_keys(self):
        config = self._base_config()
        config["llm_provider"] = "gemini_cli"
        config["gemini_cli"] = {
            "binary": "gemini",
            "model": "gemini-2.5-pro",
        }

        with self.assertRaises(SystemExit) as ctx:
            transcribe_meeting.validate_config(config, require_llm=True, llm_provider="gemini_cli")

        self.assertIn("gemini_cli.prompt_template", str(ctx.exception))


class PromptTests(unittest.TestCase):
    def test_postprocess_replaces_transcription_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("Header\n{{transcription}}\nFooter", encoding="utf-8")

            config = {
                "ollama": {
                    "model": "test-model",
                    "url": "http://localhost:11434/api/generate",
                    "prompt_template": str(prompt_file),
                }
            }

            logger = transcribe_meeting.setup_logging()
            transcript = "TEST TRANSCRIPT"

            with patch.object(transcribe_meeting, "call_ollama", return_value="ok") as mocked:
                result = transcribe_meeting.postprocess_with_provider(
                    transcript,
                    config,
                    logger,
                    llm_provider="ollama",
                )

            self.assertEqual(result, "ok")
            sent_prompt = mocked.call_args[0][0]
            self.assertIn("TEST TRANSCRIPT", sent_prompt)
            self.assertNotIn("{{transcription}}", sent_prompt)

    def test_postprocess_with_provider_routes_to_ollama(self):
        config = {
            "ollama": {
                "model": "test-model",
                "url": "http://localhost:11434/api/generate",
                "prompt_template": "/tmp/prompt.txt",
            }
        }
        logger = transcribe_meeting.setup_logging()

        with patch.object(transcribe_meeting, "build_prompt", return_value="prompt") as mocked_prompt:
            with patch.object(transcribe_meeting, "call_ollama", return_value="ok") as mocked_ollama:
                result = transcribe_meeting.postprocess_with_provider(
                    "transcript",
                    config,
                    logger,
                    llm_provider="ollama",
                )

        self.assertEqual(result, "ok")
        mocked_prompt.assert_called_once()
        mocked_ollama.assert_called_once()

    def test_postprocess_with_provider_routes_to_gemini_cli(self):
        config = {
            "gemini_cli": {
                "binary": "gemini",
                "model": "gemini-2.5-pro",
                "prompt_template": "/tmp/prompt.txt",
            }
        }
        logger = transcribe_meeting.setup_logging()

        with patch.object(transcribe_meeting, "build_prompt", return_value="prompt") as mocked_prompt:
            with patch.object(transcribe_meeting, "call_gemini_cli", return_value="ok") as mocked_gemini:
                result = transcribe_meeting.postprocess_with_provider(
                    "transcript",
                    config,
                    logger,
                    llm_provider="gemini_cli",
                )

        self.assertEqual(result, "ok")
        mocked_prompt.assert_called_once()
        mocked_gemini.assert_called_once()

    def test_call_gemini_cli_raises_on_subprocess_error(self):
        config = {
            "gemini_cli": {
                "binary": "gemini",
                "model": "gemini-2.5-pro",
                "prompt_template": "/tmp/prompt.txt",
            }
        }
        logger = transcribe_meeting.setup_logging()
        err = subprocess.CalledProcessError(
            returncode=1, cmd=["gemini"], stderr="boom"
        )
        with patch.object(transcribe_meeting.subprocess, "run", side_effect=err):
            with self.assertRaises(subprocess.CalledProcessError):
                transcribe_meeting.call_gemini_cli("prompt", config, logger)

    def test_call_gemini_cli_handles_empty_stdout(self):
        config = {
            "gemini_cli": {
                "binary": "gemini",
                "model": "gemini-2.5-pro",
                "prompt_template": "/tmp/prompt.txt",
            }
        }
        logger = transcribe_meeting.setup_logging()

        class _Resp:
            stdout = "   "

        with patch.object(transcribe_meeting.subprocess, "run", return_value=_Resp()):
            result = transcribe_meeting.call_gemini_cli("prompt", config, logger)

        self.assertEqual(result, "")

    def test_smoke_check_gemini_cli_fails_when_binary_missing(self):
        config = {"gemini_cli": {"binary": "gemini"}}
        logger = transcribe_meeting.setup_logging()

        with patch.object(transcribe_meeting.shutil, "which", return_value=None):
            with self.assertRaises(SystemExit) as ctx:
                transcribe_meeting.smoke_check_gemini_cli(config, logger)

        self.assertIn("Gemini CLI не найден", str(ctx.exception))

    def test_smoke_check_gemini_cli_passes_when_binary_found(self):
        config = {"gemini_cli": {"binary": "gemini"}}
        logger = transcribe_meeting.setup_logging()

        with patch.object(transcribe_meeting.shutil, "which", return_value="/usr/local/bin/gemini"):
            transcribe_meeting.smoke_check_gemini_cli(config, logger)

    def test_call_ollama_sends_reasoning_effort_when_configured(self):
        config = {
            "ollama": {
                "model": "test-model",
                "url": "http://localhost:11434/api/generate",
                "prompt_template": "/tmp/prompt.txt",
                "reasoning_effort": "high",
            }
        }
        logger = transcribe_meeting.setup_logging()

        class _Resp:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"response": "ok"}

        with patch.object(transcribe_meeting.requests, "post", return_value=_Resp()) as mocked_post:
            result = transcribe_meeting.call_ollama("prompt", config, logger)

        self.assertEqual(result, "ok")
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload.get("reasoning_effort"), "high")

    def test_call_ollama_retries_without_reasoning_effort_on_400(self):
        config = {
            "ollama": {
                "model": "test-model",
                "url": "http://localhost:11434/api/generate",
                "prompt_template": "/tmp/prompt.txt",
                "reasoning_effort": "high",
            }
        }
        logger = transcribe_meeting.setup_logging()

        class _Resp400:
            status_code = 400

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"response": ""}

        class _Resp200:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"response": "ok-after-retry"}

        captured_payloads = []

        def _post_side_effect(*_, **kwargs):
            captured_payloads.append(dict(kwargs["json"]))
            if len(captured_payloads) == 1:
                return _Resp400()
            return _Resp200()

        with patch.object(
            transcribe_meeting.requests,
            "post",
            side_effect=_post_side_effect,
        ) as mocked_post:
            result = transcribe_meeting.call_ollama("prompt", config, logger)

        self.assertEqual(result, "ok-after-retry")
        self.assertEqual(mocked_post.call_count, 2)
        first_payload = captured_payloads[0]
        second_payload = captured_payloads[1]
        self.assertEqual(first_payload.get("reasoning_effort"), "high")
        self.assertNotIn("reasoning_effort", second_payload)


class PipelineUtilityTests(unittest.TestCase):
    def test_parse_csv_arg(self):
        value = " one, two ,,three "
        self.assertEqual(transcribe_meeting.parse_csv_arg(value), ["one", "two", "three"])

    def test_should_run_stage_range(self):
        self.assertTrue(transcribe_meeting.should_run_stage("transcribe", "convert", "llm"))
        self.assertFalse(transcribe_meeting.should_run_stage("update", "convert", "llm"))

    def test_make_safe_suffix(self):
        self.assertEqual(
            transcribe_meeting.make_safe_suffix("gpt/oss:120b cloud"),
            "gpt_oss_120b_cloud",
        )

    def test_save_transcript_output_creates_accessible_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes_dir = Path(tmp) / "notes"
            logger = transcribe_meeting.setup_logging()
            transcript = "raw transcript text"

            result_path = transcribe_meeting.save_transcript_output(
                transcript=transcript,
                base_name="meeting-1",
                notes_dir=notes_dir,
                logger=logger,
            )

            self.assertTrue(result_path.exists())
            self.assertEqual(result_path.name, "meeting-1.transcript.txt")
            self.assertEqual(result_path.read_text(encoding="utf-8"), transcript)


if __name__ == "__main__":
    unittest.main()
