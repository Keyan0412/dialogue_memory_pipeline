from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialogue_memory_pipeline import load_sample_dialogue
from dialogue_memory_pipeline.core.schemas import Utterance
from dialogue_memory_pipeline.modules.state_extractor import LLMStateExtractor
from dialogue_memory_pipeline.modules.transition_judge import _coerce_bool


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def complete_json(self, system_prompt: str, user_prompt: str):
        return self.response


class PackageLayoutTests(unittest.TestCase):
    def test_packaged_sample_dialogue_can_be_loaded(self) -> None:
        utterances = load_sample_dialogue()
        self.assertGreater(len(utterances), 1)
        self.assertEqual(utterances[0].turn_id, 0)

    def test_state_extractor_rejects_misaligned_turn_ids(self) -> None:
        extractor = LLMStateExtractor(
            FakeLLM(
                [
                    {
                        "turn_id": 1,
                        "speaker": "user",
                        "summary_topic": "topic",
                        "intent": "ask",
                        "salient_entities": [],
                        "cue_markers": [],
                        "obligation": {"opens": [], "resolves": []},
                    }
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "turn_ids do not match"):
            extractor.extract(
                [
                    Utterance(turn_id=0, speaker="user", text="hello"),
                ]
            )

    def test_transition_judge_bool_coercion_handles_string_false(self) -> None:
        self.assertFalse(_coerce_bool("false", default=True))
        self.assertTrue(_coerce_bool("true", default=False))


if __name__ == "__main__":
    unittest.main()
