from __future__ import annotations

from abc import ABC, abstractmethod
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

    def build(self, segments: Sequence[Segment]) -> List[EpisodeMemory]:
        out: List[EpisodeMemory] = []
        for seg in segments:
            system_prompt = MEMORY_BUILDER_PROMPT
            user_prompt = (
                f"Segment state: {seg.segment_state.to_dict()}\n"
                f"Utterances: {[{'turn_id': u.turn_id, 'speaker': u.speaker, 'text': u.text} for u in seg.utterances]}\n"
                f"Local states: {[compact_local_state_payload(ls) for ls in seg.local_states]}"
            )
            obj = self.llm.complete_json(system_prompt, user_prompt)
            importance = int(obj.get("importance", 1))
            importance = max(1, min(5, importance))
            out.append(
                EpisodeMemory(
                    episode_id=seg.segment_id.replace("seg", "ep"),
                    utterance_span=[seg.utterances[0].turn_id, seg.utterances[-1].turn_id],
                    utterances=list(seg.utterances),
                    retrieval_summary_zh=str(obj["retrieval_summary_zh"]),
                    retrieval_summary_en=str(obj["retrieval_summary_en"]),
                    key_entities_zh=self._normalize_entities(obj.get("key_entities_zh", [])),
                    key_entities_en=self._normalize_entities(obj.get("key_entities_en", [])),
                    importance=importance,
                )
            )
        return out
