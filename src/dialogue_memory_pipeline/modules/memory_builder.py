from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import List, Sequence

from dialogue_memory_pipeline.clients.llm_client import JSONLLM
from dialogue_memory_pipeline.core.schemas import EpisodeMemory, Segment
from dialogue_memory_pipeline.modules.state_extractor import compact_local_state_payload
from dialogue_memory_pipeline.prompts import MEMORY_BUILDER_PROMPT


class BaseMemoryBuilder(ABC):
    @abstractmethod
    def build(self, segments: Sequence[Segment]) -> List[EpisodeMemory]:
        raise NotImplementedError


class LLMMemoryBuilder(BaseMemoryBuilder):
    def __init__(self, llm: JSONLLM) -> None:
        self.llm = llm

    def _normalize_entities(self, value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def build(self, segments: Sequence[Segment], dialogue_id: str | None = None) -> List[EpisodeMemory]:
        out: List[EpisodeMemory] = []
        episode_count = len(segments)
        for index, seg in enumerate(segments):
            system_prompt = MEMORY_BUILDER_PROMPT
            user_prompt = (
                f"Segment state: {seg.segment_state.to_dict()}\n"
                f"Utterances: {[{'turn_id': u.turn_id, 'speaker': u.speaker, 'text': u.text} for u in seg.utterances]}\n"
                f"Local states: {[compact_local_state_payload(ls) for ls in seg.local_states]}"
            )
            obj = self.llm.complete_json(system_prompt, user_prompt)
            importance = int(obj.get("importance", 1))
            importance = max(1, min(5, importance))
            turn_start = seg.utterances[0].turn_id
            turn_end = seg.utterances[-1].turn_id
            episode_suffix = f"ep_{index:03d}"
            episode_id = f"{dialogue_id}:{episode_suffix}" if dialogue_id else episode_suffix
            out.append(
                EpisodeMemory(
                    dialogue_id=dialogue_id,
                    episode_id=episode_id,
                    segment_id=seg.segment_id,
                    episode_index=index,
                    episode_count=episode_count,
                    utterance_span=[turn_start, turn_end],
                    turn_start=turn_start,
                    turn_end=turn_end,
                    utterance_count=len(seg.utterances),
                    relative_start=round(index / episode_count, 6) if episode_count else 0.0,
                    relative_end=round((index + 1) / episode_count, 6) if episode_count else 1.0,
                    utterances=list(seg.utterances),
                    stable_topic=seg.segment_state.stable_topic,
                    discourse_goal=seg.segment_state.discourse_goal,
                    open_obligations=list(seg.segment_state.open_obligations),
                    retrieval_summary_zh=str(obj["retrieval_summary_zh"]),
                    retrieval_summary_en=str(obj["retrieval_summary_en"]),
                    key_entities_zh=self._normalize_entities(obj.get("key_entities_zh", [])),
                    key_entities_en=self._normalize_entities(obj.get("key_entities_en", [])),
                    importance=importance,
                    token_estimate=_estimate_token_count(seg),
                )
            )
        return out


def _estimate_token_count(segment: Segment) -> int:
    text = "\n".join(utterance.text for utterance in segment.utterances)
    if not text:
        return 0
    # Keep this cheap and deterministic for upstream export metadata.
    return max(1, math.ceil(len(text) / 4))
