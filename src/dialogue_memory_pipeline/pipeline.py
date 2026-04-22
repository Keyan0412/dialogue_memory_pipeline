from __future__ import annotations

import json
import os
import time
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dialogue_memory_pipeline.clients.llm_client import JSONLLM, OpenAIJSONLLM
from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.core.schemas import CandidateBoundary, LocalState, Segment, Utterance
from dialogue_memory_pipeline.modules.candidate_generator import CandidateBoundaryGenerator
from dialogue_memory_pipeline.modules.memory_builder import LLMMemoryBuilder
from dialogue_memory_pipeline.modules.segment_state import aggregate_segment_state
from dialogue_memory_pipeline.modules.state_extractor import LLMStateExtractor
from dialogue_memory_pipeline.modules.transition_judge import LLMTransitionJudge


class DialogueSegmentationPipeline:
    def __init__(
        self,
        llm: JSONLLM,
        config: PipelineConfig | None = None,
        candidate_llm: JSONLLM | None = None,
        local_state_llm: JSONLLM | None = None,
        transition_llm: JSONLLM | None = None,
        memory_llm: JSONLLM | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.candidate_generator = CandidateBoundaryGenerator(candidate_llm or llm, self.config)
        self.local_state_extractor = LLMStateExtractor(local_state_llm or llm, self.config)
        self.transition_judge = LLMTransitionJudge(transition_llm or llm, self.config)
        self.memory_builder = LLMMemoryBuilder(memory_llm or llm)

    @classmethod
    def from_openai(
        cls,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        config: PipelineConfig | None = None,
        candidate_model: str | None = None,
        local_state_model: str | None = None,
        transition_model: str | None = None,
        memory_model: str | None = None,
    ) -> "DialogueSegmentationPipeline":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        resolved_base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL")
        effective_config = config or PipelineConfig()

        default_llm = OpenAIJSONLLM(
            api_key=resolved_api_key,
            model=model,
            base_url=resolved_base_url,
        )
        candidate_llm: JSONLLM | None = None
        local_state_llm: JSONLLM | None = None
        transition_llm: JSONLLM | None = None
        memory_llm: JSONLLM | None = None

        if candidate_model and candidate_model != model:
            candidate_llm = OpenAIJSONLLM(
                api_key=resolved_api_key,
                model=candidate_model,
                base_url=resolved_base_url,
            )
        if effective_config.local_state_transport == "bailian_batch_chat":
            batch_base_url = _resolve_bailian_batch_base_url(resolved_base_url)
            local_state_llm = OpenAIJSONLLM(
                api_key=resolved_api_key,
                model=local_state_model or model,
                base_url=batch_base_url,
                api_mode="chat_completions",
                timeout=1800.0,
            )
        elif local_state_model and local_state_model != model:
            local_state_llm = OpenAIJSONLLM(
                api_key=resolved_api_key,
                model=local_state_model,
                base_url=resolved_base_url,
            )
        if transition_model and transition_model != model:
            transition_llm = OpenAIJSONLLM(
                api_key=resolved_api_key,
                model=transition_model,
                base_url=resolved_base_url,
            )
        if memory_model and memory_model != model:
            memory_llm = OpenAIJSONLLM(
                api_key=resolved_api_key,
                model=memory_model,
                base_url=resolved_base_url,
            )
        return cls(
            llm=default_llm,
            config=effective_config,
            candidate_llm=candidate_llm,
            local_state_llm=local_state_llm,
            transition_llm=transition_llm,
            memory_llm=memory_llm,
        )

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        config: PipelineConfig | None = None,
    ) -> "DialogueSegmentationPipeline":
        resolved_model = model or os.getenv("OPENAI_MODEL", "qwen3.5-plus")
        candidate_model = os.getenv("OPENAI_MODEL_CANDIDATE", resolved_model)
        local_state_model = os.getenv("OPENAI_MODEL_LOCAL_STATE", resolved_model)
        transition_model = os.getenv("OPENAI_MODEL_TRANSITION", resolved_model)
        memory_model = os.getenv("OPENAI_MODEL_MEMORY", resolved_model)
        return cls.from_openai(
            model=resolved_model,
            config=config,
            candidate_model=candidate_model,
            local_state_model=local_state_model,
            transition_model=transition_model,
            memory_model=memory_model,
        )

    def run(
        self,
        utterances: Sequence[Utterance],
        dialogue_id: str | None = None,
    ) -> Dict[str, object]:
        print("[pipeline] Starting pipeline run")
        print(f"[pipeline] Input utterances: {len(utterances)}")
        if dialogue_id:
            print(f"[pipeline] Dialogue id: {dialogue_id}")

        print("[pipeline] Stage 1/4: selecting final candidate boundaries")
        started = time.perf_counter()
        candidates = self.candidate_generator.generate(utterances)
        candidate_selection_seconds = time.perf_counter() - started
        print(f"[pipeline] Stage 1/4 done in {candidate_selection_seconds:.2f}s")
        print(f"[pipeline] Final candidates kept: {len(candidates)}")

        print("[pipeline] Stage 2/4: extracting local states")
        started = time.perf_counter()
        local_states = self.local_state_extractor.extract(utterances)
        state_extraction_seconds = time.perf_counter() - started
        print(f"[pipeline] Stage 2/4 done in {state_extraction_seconds:.2f}s")
        print(f"[pipeline] Local states extracted: {len(local_states)}")

        print("[pipeline] Stage 3/4: segmenting dialogue")
        started = time.perf_counter()
        segments, decisions = self._segment(utterances, local_states, candidates)
        raw_segment_count = len(segments)
        segments = self._cleanup_merge(segments)
        segmentation_seconds = time.perf_counter() - started
        print(f"[pipeline] Stage 3/4 done in {segmentation_seconds:.2f}s")
        print(f"[pipeline] Segments before cleanup: {raw_segment_count}")
        print(f"[pipeline] Segments after cleanup: {len(segments)}")

        print("[pipeline] Stage 4/4: building episodic memories")
        started = time.perf_counter()
        episodes = self.memory_builder.build(segments, dialogue_id=dialogue_id)
        memory_building_seconds = time.perf_counter() - started
        print(f"[pipeline] Stage 4/4 done in {memory_building_seconds:.2f}s")
        print(f"[pipeline] Episodes built: {len(episodes)}")
        print("[pipeline] Pipeline run complete")

        return {
            "dialogue_id": dialogue_id,
            "candidates": [c.to_dict() for c in candidates],
            "local_states": [ls.to_dict() for ls in local_states],
            "decisions": [x for x in decisions],
            "segments": [s.to_dict() for s in segments],
            "episodes": [e.to_dict() for e in episodes],
            "timing": {
                "candidate_selection_seconds": candidate_selection_seconds,
                "state_extraction_seconds": state_extraction_seconds,
                "segmentation_seconds": segmentation_seconds,
                "memory_building_seconds": memory_building_seconds,
            },
        }

    @staticmethod
    def extract_memories(result: Dict[str, object], language: str | None = None) -> List[Dict[str, Any]]:
        """Extract compact memory records from a pipeline result."""
        episodes = result.get("episodes", [])
        if not isinstance(episodes, list):
            raise ValueError("result['episodes'] must be a list")

        if language is None:
            return [
                {
                    "dialogue_id": episode.get("dialogue_id"),
                    "episode_id": episode["episode_id"],
                    "utterance_span": episode["utterance_span"],
                    "retrieval_summary_zh": episode["retrieval_summary_zh"],
                    "retrieval_summary_en": episode["retrieval_summary_en"],
                    "key_entities_zh": episode["key_entities_zh"],
                    "key_entities_en": episode["key_entities_en"],
                    "importance": episode["importance"],
                }
                for episode in episodes
            ]

        if language not in {"zh", "en"}:
            raise ValueError("language must be 'zh', 'en', or None")

        return [
            {
                "dialogue_id": episode.get("dialogue_id"),
                "episode_id": episode["episode_id"],
                "utterance_span": episode["utterance_span"],
                "retrieval_summary": episode[f"retrieval_summary_{language}"],
                "key_entities": episode[f"key_entities_{language}"],
                "importance": episode["importance"],
            }
            for episode in episodes
        ]

    @staticmethod
    def normalize_episode_records(
        result: Dict[str, object],
        dialogue_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return normalized episode export records suitable for downstream retrieval."""
        episodes = result.get("episodes", [])
        if not isinstance(episodes, list):
            raise ValueError("result['episodes'] must be a list")

        resolved_dialogue_id = dialogue_id if dialogue_id is not None else result.get("dialogue_id")
        normalized: List[Dict[str, Any]] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                raise ValueError("Each episode must be a dictionary")

            turn_start = int(episode.get("turn_start", episode["utterance_span"][0]))
            turn_end = int(episode.get("turn_end", episode["utterance_span"][-1]))
            utterances = episode.get("utterances", [])
            stable_topic = str(episode.get("stable_topic", ""))
            discourse_goal = str(episode.get("discourse_goal", ""))
            open_obligations = list(episode.get("open_obligations", []))
            record_dialogue_id = episode.get("dialogue_id", resolved_dialogue_id)
            episode_id = str(episode.get("episode_id", f"ep_{index:03d}"))
            if record_dialogue_id and ":" not in episode_id:
                episode_id = f"{record_dialogue_id}:{episode_id}"

            normalized.append(
                {
                    "dialogue_id": record_dialogue_id,
                    "episode_id": episode_id,
                    "segment_id": episode.get("segment_id"),
                    "episode_index": int(episode.get("episode_index", index)),
                    "episode_count": int(episode.get("episode_count", len(episodes))),
                    "turn_start": turn_start,
                    "turn_end": turn_end,
                    "utterance_count": int(episode.get("utterance_count", len(utterances))),
                    "relative_start": float(episode.get("relative_start", 0.0)),
                    "relative_end": float(episode.get("relative_end", 1.0)),
                    "stable_topic": stable_topic,
                    "discourse_goal": discourse_goal,
                    "open_obligations": open_obligations,
                    "retrieval_summary_zh": str(episode["retrieval_summary_zh"]),
                    "retrieval_summary_en": str(episode["retrieval_summary_en"]),
                    "key_entities_zh": list(episode.get("key_entities_zh", [])),
                    "key_entities_en": list(episode.get("key_entities_en", [])),
                    "importance": int(episode.get("importance", 1)),
                    "token_estimate": int(episode.get("token_estimate", 0)),
                }
            )
        return normalized

    @staticmethod
    def export_episodes(
        result: Dict[str, object],
        path: str | Path,
        dialogue_id: str | None = None,
    ) -> int:
        """Write normalized episode records as JSONL and return the row count."""
        records = DialogueSegmentationPipeline.normalize_episode_records(result, dialogue_id=dialogue_id)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        return len(records)

    def _segment(
        self,
        utterances: Sequence[Utterance],
        local_states: Sequence[LocalState],
        candidates: Sequence[CandidateBoundary],
    ) -> Tuple[List[Segment], List[Dict[str, object]]]:
        """Convert boundary candidates into finalized segments.

        This method walks through the candidate boundaries in chronological
        order and treats each one as a possible split point. For every
        candidate:

        1. It builds the current left segment from `current_start` through the
           candidate boundary.
        2. It skips candidates that would create a left segment shorter than
           `min_segment_len`.
        3. It aggregates the left segment into a `SegmentState`.
        4. It collects a short preview of local states on the right side, using
           `right_preview_window`, so the transition judge can inspect what the
           upcoming dialogue looks like without committing to a split yet.
        5. It asks the transition judge whether the right side should start a
           new segment, given the left segment state and the candidate
           information.
        6. It records the full decision trace in `decisions_log` for later
           inspection.

        If the judge decides to split, the left span becomes a finalized
        `Segment`, `current_start` moves to the next turn, and later candidates
        are evaluated relative to this new segment start.

        After all candidates are processed, any remaining trailing utterances
        from `current_start` to the end of the dialogue are emitted as the last
        segment.

        Returns:
            A tuple of:
            - the ordered list of segments produced before cleanup/merge
            - a per-candidate decision log containing the candidate payload,
              left segment state, right preview turns, and judge output
        """
        candidate_map = {c.boundary_after_turn: c for c in candidates}
        boundaries = sorted(candidate_map.keys())

        current_start = 0
        segments: List[Segment] = []
        decisions_log: List[Dict[str, object]] = []

        for boundary_after_turn in boundaries:
            idx = boundary_after_turn
            if idx - current_start + 1 < self.config.min_segment_len:
                continue
            
            # Build the left segment state
            left_utts = utterances[current_start : idx + 1]
            left_states = local_states[current_start : idx + 1]
            left_segment_state = aggregate_segment_state(left_utts, left_states)

            # Build the right preview window
            right_start = idx + 1
            right_end = min(len(utterances), right_start + self.config.right_preview_window)
            right_states = local_states[right_start:right_end]
            if not right_states:
                continue
            
            # call transition judge
            cand = candidate_map[boundary_after_turn]
            decision = self.transition_judge.judge(
                left_state=left_segment_state,
                right_local_states=right_states,
                candidate_features=cand.to_dict(),
            )
            decisions_log.append(
                {
                    "boundary_after_turn": boundary_after_turn,
                    "candidate": cand.to_dict(),
                    "left_state": left_segment_state.to_dict(),
                    "right_preview_turns": [ls.turn_id for ls in right_states],
                    "decision": decision.to_dict(),
                }
            )

            if decision.should_split:
                seg_id = f"seg_{len(segments):03d}"
                seg_utts = list(utterances[current_start : idx + 1])
                seg_states = list(local_states[current_start : idx + 1])
                segments.append(
                    Segment(
                        segment_id=seg_id,
                        utterances=seg_utts,
                        local_states=seg_states,
                        segment_state=aggregate_segment_state(seg_utts, seg_states),
                    )
                )
                current_start = idx + 1

        # Handle any remaining utterances after the last boundary
        if current_start < len(utterances):
            seg_id = f"seg_{len(segments):03d}"
            seg_utts = list(utterances[current_start:])
            seg_states = list(local_states[current_start:])
            segments.append(
                Segment(
                    segment_id=seg_id,
                    utterances=seg_utts,
                    local_states=seg_states,
                    segment_state=aggregate_segment_state(seg_utts, seg_states),
                )
            )

        return segments, decisions_log

    def _cleanup_merge(self, segments: Sequence[Segment]) -> List[Segment]:
        """Merge only segments that are too short to stand on their own.

        The current policy is intentionally conservative: adjacent segments are
        kept separate unless the later segment has fewer utterances than
        `min_segment_len`, in which case it is merged into the previous one.
        """
        if not segments:
            return []
        merged: List[Segment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            too_short = len(seg.utterances) < self.config.min_segment_len
            if too_short:
                new_utts = prev.utterances + seg.utterances
                new_states = prev.local_states + seg.local_states
                merged[-1] = Segment(
                    segment_id=prev.segment_id,
                    utterances=new_utts,
                    local_states=new_states,
                    segment_state=aggregate_segment_state(new_utts, new_states),
                )
            else:
                merged.append(seg)
        return merged



def load_dialogue(path: str | Path) -> List[Utterance]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Utterance(turn_id=int(x["turn_id"]), speaker=x["speaker"], text=x["text"]) for x in items]


def load_sample_dialogue() -> List[Utterance]:
    data_path = files("dialogue_memory_pipeline").joinpath("data", "sample_dialogue.json")
    items = json.loads(data_path.read_text(encoding="utf-8"))
    return [Utterance(turn_id=int(x["turn_id"]), speaker=x["speaker"], text=x["text"]) for x in items]


def _resolve_bailian_batch_base_url(base_url: str | None) -> str:
    if not base_url:
        return "https://batch.dashscope.aliyuncs.com/compatible-mode/v1"
    if "dashscope-intl.aliyuncs.com" in base_url:
        return "https://batch.dashscope-intl.aliyuncs.com/compatible-mode/v1"
    return "https://batch.dashscope.aliyuncs.com/compatible-mode/v1"
