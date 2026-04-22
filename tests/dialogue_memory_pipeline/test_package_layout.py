from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialogue_memory_pipeline import load_sample_dialogue
from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.core.schemas import LocalState, ObligationState, Segment, SegmentState, Utterance
from dialogue_memory_pipeline.modules.memory_builder import LLMMemoryBuilder
from dialogue_memory_pipeline.pipeline import DialogueSegmentationPipeline
from dialogue_memory_pipeline.modules.state_extractor import LLMStateExtractor
from dialogue_memory_pipeline.modules.transition_judge import _coerce_bool


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def complete_json(self, system_prompt: str, user_prompt: str):
        return self.response


class EchoStateLLM:
    def complete_json(self, system_prompt: str, user_prompt: str):
        payload = json.loads(user_prompt)
        return [
            {
                "turn_id": item["turn_id"],
                "speaker": item["speaker"],
                "summary_topic": f"topic-{item['turn_id']}",
                "intent": "describe_problem",
                "obligation": {"opens": [], "resolves": []},
            }
            for item in payload
        ]


class RecordingLLM:
    instances: list["RecordingLLM"] = []

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None, api_mode: str = "responses", timeout: float | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.api_mode = api_mode
        self.timeout = timeout
        RecordingLLM.instances.append(self)

    def complete_json(self, system_prompt: str, user_prompt: str):
        return {}


class CandidateLLM:
    def complete_json(self, system_prompt: str, user_prompt: str):
        return [{"boundary_after_turn": 0, "score": 0.95, "reasoning": "clear topic shift"}]


class StateLLM:
    def complete_json(self, system_prompt: str, user_prompt: str):
        payload = json.loads(user_prompt)
        return [
            {
                "turn_id": item["turn_id"],
                "speaker": item["speaker"],
                "summary_topic": "travel booking" if item["turn_id"] == 0 else "hotel booking",
                "intent": "ask" if item["turn_id"] == 0 else "change_topic",
                "salient_entities": ["flight"] if item["turn_id"] == 0 else ["hotel"],
                "cue_markers": [],
                "obligation": {"opens": ["provide booking details"] if item["turn_id"] == 0 else [], "resolves": []},
            }
            for item in payload
        ]


class TransitionLLM:
    def complete_json(self, system_prompt: str, user_prompt: str):
        return {
            "transition": "shift_new_topic",
            "confidence": 0.9,
            "should_split": True,
            "reasoning": {"signal": "topic change"},
        }


class MemoryLLM:
    def complete_json(self, system_prompt: str, user_prompt: str):
        if "hotel booking" in user_prompt:
            return {
                "retrieval_summary_zh": "转向预订酒店并确认房型偏好。",
                "retrieval_summary_en": "Shifted to booking a hotel and confirming room preferences.",
                "key_entities_zh": ["酒店预订", "房型偏好"],
                "key_entities_en": ["hotel booking", "room preferences"],
                "importance": 3,
            }
        return {
            "retrieval_summary_zh": "先讨论航班改签与订单信息。",
            "retrieval_summary_en": "Discussed flight rescheduling and booking details first.",
            "key_entities_zh": ["航班改签", "订单信息"],
            "key_entities_en": ["flight rescheduling", "booking details"],
            "importance": 4,
        }


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

    def test_state_extractor_accepts_minimal_schema(self) -> None:
        extractor = LLMStateExtractor(
            FakeLLM(
                [
                    {
                        "turn_id": 0,
                        "speaker": "user",
                        "summary_topic": "topic",
                        "intent": "describe_problem",
                        "obligation": {"opens": [], "resolves": []},
                    }
                ]
            )
        )

        result = extractor.extract([Utterance(turn_id=0, speaker="user", text="hello")])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].summary_topic, "topic")
        self.assertEqual(result[0].intent, "describe_problem")
        self.assertEqual(result[0].salient_entities, [])
        self.assertEqual(result[0].cue_markers, [])
        self.assertEqual(result[0].obligation.opens, [])
        self.assertEqual(result[0].obligation.resolves, [])

    def test_state_extractor_supports_chunked_parallel_execution(self) -> None:
        extractor = LLMStateExtractor(
            EchoStateLLM(),
            PipelineConfig(local_state_chunk_size=2, local_state_max_parallel=2),
        )

        result = extractor.extract(
            [
                Utterance(turn_id=0, speaker="user", text="a"),
                Utterance(turn_id=1, speaker="assistant", text="b"),
                Utterance(turn_id=2, speaker="user", text="c"),
                Utterance(turn_id=3, speaker="assistant", text="d"),
                Utterance(turn_id=4, speaker="user", text="e"),
            ]
        )

        self.assertEqual([state.turn_id for state in result], [0, 1, 2, 3, 4])
        self.assertEqual([state.summary_topic for state in result], ["topic-0", "topic-1", "topic-2", "topic-3", "topic-4"])

    def test_transition_judge_bool_coercion_handles_string_false(self) -> None:
        self.assertFalse(_coerce_bool("false", default=True))
        self.assertTrue(_coerce_bool("true", default=False))

    def test_memory_builder_outputs_bilingual_fields(self) -> None:
        builder = LLMMemoryBuilder(
            FakeLLM(
                {
                    "retrieval_summary_zh": "围绕双语记忆输出更新了摘要与实体结构。",
                    "retrieval_summary_en": "Updated the summary and entity structure for bilingual memory output.",
                    "key_entities_zh": ["双语记忆", "摘要结构"],
                    "key_entities_en": ["bilingual memory", "summary structure"],
                    "importance": 4,
                }
            )
        )

        segment = Segment(
            segment_id="seg_000",
            utterances=[Utterance(turn_id=0, speaker="user", text="please update the memory builder")],
            local_states=[
                LocalState(
                    turn_id=0,
                    speaker="user",
                    summary_topic="memory builder update",
                    intent="request change",
                    salient_entities=["memory builder"],
                    cue_markers=[],
                    obligation=ObligationState(),
                )
            ],
            segment_state=SegmentState(stable_topic="memory builder", discourse_goal="update bilingual output"),
        )

        episodes = builder.build([segment])

        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertIsNone(episode.dialogue_id)
        self.assertEqual(episode.segment_id, "seg_000")
        self.assertEqual(episode.episode_index, 0)
        self.assertEqual(episode.episode_count, 1)
        self.assertEqual(episode.turn_start, 0)
        self.assertEqual(episode.turn_end, 0)
        self.assertEqual(episode.utterance_count, 1)
        self.assertEqual(episode.relative_start, 0.0)
        self.assertEqual(episode.relative_end, 1.0)
        self.assertEqual(episode.stable_topic, "memory builder")
        self.assertEqual(episode.discourse_goal, "update bilingual output")
        self.assertEqual(episode.open_obligations, [])
        self.assertEqual(episode.retrieval_summary_zh, "围绕双语记忆输出更新了摘要与实体结构。")
        self.assertEqual(episode.retrieval_summary_en, "Updated the summary and entity structure for bilingual memory output.")
        self.assertEqual(episode.key_entities_zh, ["双语记忆", "摘要结构"])
        self.assertEqual(episode.key_entities_en, ["bilingual memory", "summary structure"])
        self.assertEqual(episode.retrieval_summary, episode.retrieval_summary_en)
        self.assertEqual(episode.key_entities, episode.key_entities_en)
        self.assertGreaterEqual(episode.token_estimate, 1)

    def test_extract_memories_supports_bilingual_and_single_language_views(self) -> None:
        result = {
            "episodes": [
                {
                    "dialogue_id": None,
                    "episode_id": "ep_000",
                    "utterance_span": [0, 2],
                    "retrieval_summary_zh": "讨论双语记忆抽取。",
                    "retrieval_summary_en": "Discussed bilingual memory extraction.",
                    "key_entities_zh": ["双语记忆", "抽取"],
                    "key_entities_en": ["bilingual memory", "extraction"],
                    "importance": 3,
                }
            ]
        }

        self.assertEqual(
            DialogueSegmentationPipeline.extract_memories(result),
            [
                {
                    "dialogue_id": None,
                    "episode_id": "ep_000",
                    "utterance_span": [0, 2],
                    "retrieval_summary_zh": "讨论双语记忆抽取。",
                    "retrieval_summary_en": "Discussed bilingual memory extraction.",
                    "key_entities_zh": ["双语记忆", "抽取"],
                    "key_entities_en": ["bilingual memory", "extraction"],
                    "importance": 3,
                }
            ],
        )
        self.assertEqual(
            DialogueSegmentationPipeline.extract_memories(result, language="en"),
            [
                {
                    "dialogue_id": None,
                    "episode_id": "ep_000",
                    "utterance_span": [0, 2],
                    "retrieval_summary": "Discussed bilingual memory extraction.",
                    "key_entities": ["bilingual memory", "extraction"],
                    "importance": 3,
                }
            ],
        )
        self.assertEqual(
            DialogueSegmentationPipeline.extract_memories(result, language="zh"),
            [
                {
                    "dialogue_id": None,
                    "episode_id": "ep_000",
                    "utterance_span": [0, 2],
                    "retrieval_summary": "讨论双语记忆抽取。",
                    "key_entities": ["双语记忆", "抽取"],
                    "importance": 3,
                }
            ],
        )

    def test_from_env_supports_stage_specific_models(self) -> None:
        RecordingLLM.instances.clear()
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "OPENAI_MODEL": "default-model",
            "OPENAI_MODEL_CANDIDATE": "candidate-model",
            "OPENAI_MODEL_LOCAL_STATE": "local-model",
            "OPENAI_MODEL_TRANSITION": "transition-model",
            "OPENAI_MODEL_MEMORY": "memory-model",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("dialogue_memory_pipeline.pipeline.OpenAIJSONLLM", RecordingLLM):
                DialogueSegmentationPipeline.from_env()

        models = [instance.model for instance in RecordingLLM.instances]
        self.assertEqual(
            models,
            ["default-model", "candidate-model", "local-model", "transition-model", "memory-model"],
        )

    def test_run_and_export_include_dialogue_metadata(self) -> None:
        pipeline = DialogueSegmentationPipeline(
            llm=FakeLLM({}),
            config=PipelineConfig(top_p_candidates=1.0, min_candidate_score=0.0, min_segment_len=1),
            candidate_llm=CandidateLLM(),
            local_state_llm=StateLLM(),
            transition_llm=TransitionLLM(),
            memory_llm=MemoryLLM(),
        )

        result = pipeline.run(
            [
                Utterance(turn_id=0, speaker="user", text="I need to change my flight."),
                Utterance(turn_id=1, speaker="assistant", text="Sure, what is your booking number?"),
            ],
            dialogue_id="dlg_test",
        )

        self.assertEqual(result["dialogue_id"], "dlg_test")
        episodes = result["episodes"]
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["dialogue_id"], "dlg_test")
        self.assertEqual(episodes[0]["episode_id"], "dlg_test:ep_000")
        self.assertEqual(episodes[1]["episode_id"], "dlg_test:ep_001")
        self.assertEqual(episodes[0]["turn_start"], 0)
        self.assertEqual(episodes[1]["turn_end"], 1)
        self.assertEqual(episodes[0]["segment_id"], "seg_000")
        self.assertEqual(episodes[1]["segment_id"], "seg_001")
        self.assertEqual(episodes[0]["stable_topic"], "travel booking")
        self.assertIn("provide booking details", episodes[0]["open_obligations"])
        self.assertGreaterEqual(episodes[0]["token_estimate"], 1)

        normalized = DialogueSegmentationPipeline.normalize_episode_records(result)
        self.assertEqual(normalized[0]["dialogue_id"], "dlg_test")
        self.assertEqual(normalized[0]["episode_id"], "dlg_test:ep_000")
        self.assertEqual(normalized[1]["episode_count"], 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "episodes.jsonl"
            row_count = DialogueSegmentationPipeline.export_episodes(result, output_path)
            self.assertEqual(row_count, 2)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            self.assertEqual(first_record["dialogue_id"], "dlg_test")
            self.assertEqual(first_record["episode_id"], "dlg_test:ep_000")
            self.assertEqual(first_record["turn_start"], 0)
            self.assertEqual(first_record["stable_topic"], "travel booking")


if __name__ == "__main__":
    unittest.main()
