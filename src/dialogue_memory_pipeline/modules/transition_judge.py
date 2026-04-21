from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from dialogue_memory_pipeline.clients.llm_client import JSONLLM
from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.core.schemas import LocalState, SegmentState, TransitionDecision
from dialogue_memory_pipeline.modules.state_extractor import compact_local_state_payload


class BaseTransitionJudge(ABC):
    @abstractmethod
    def judge(
        self,
        left_state: SegmentState,
        right_local_states: Sequence[LocalState],
        candidate_features: Dict[str, float],
    ) -> TransitionDecision:
        raise NotImplementedError


class LLMTransitionJudge(BaseTransitionJudge):
    def __init__(self, llm: JSONLLM, config: PipelineConfig) -> None:
        self.llm = llm
        self.config = config

    def judge(
        self,
        left_state: SegmentState,
        right_local_states: Sequence[LocalState],
        candidate_features: Dict[str, float],
    ) -> TransitionDecision:
        system_prompt = (
            "You judge dialogue state transitions. Output JSON with keys: transition, confidence, should_split, reasoning. "
            "transition must be one of continue_same_topic, elaborate_same_topic, respond_same_topic, introduce_related_topic, shift_new_topic."
        )
        user_prompt = (
            f"Left segment state: {left_state.to_dict()}\n"
            f"Right local states: {[compact_local_state_payload(x) for x in right_local_states]}\n"
            f"Candidate features: {candidate_features}\n"
            "Decide whether the right side still fits the current stable topic and discourse goal, or opens a new topic."
        )
        obj = self.llm.complete_json(system_prompt, user_prompt)
        transition = str(obj["transition"])
        confidence = float(obj.get("confidence", 0.6))
        should_split = _coerce_bool(obj.get("should_split"), default=transition == "shift_new_topic")
        reasoning = obj.get("reasoning", {})
        return TransitionDecision(
            transition=transition,
            confidence=confidence,
            should_split=should_split,
            reasoning=reasoning,
        )


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)
