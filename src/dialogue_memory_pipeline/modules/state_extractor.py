from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List, Sequence

from dialogue_memory_pipeline.clients.llm_client import JSONLLM
from dialogue_memory_pipeline.core.schemas import LocalState, ObligationState, Utterance
from dialogue_memory_pipeline.prompts import CHINESE_EXTRACTION_SYSTEM_PROMPT


class BaseLocalStateExtractor(ABC):
    @abstractmethod
    def extract(self, utterances: Sequence[Utterance]) -> List[LocalState]:
        raise NotImplementedError


class LLMStateExtractor(BaseLocalStateExtractor):
    def __init__(self, llm: JSONLLM) -> None:
        self.llm = llm

    def extract(self, utterances: Sequence[Utterance]) -> List[LocalState]:
        system_prompt = CHINESE_EXTRACTION_SYSTEM_PROMPT
        if not utterances:
            return []
        payload = [
            {
                "turn_id": u.turn_id,
                "speaker": u.speaker,
                "text": u.text,
            }
            for u in utterances
        ]
        obj = self.llm.complete_json(system_prompt, json.dumps(payload, ensure_ascii=False))
        items = obj["items"] if isinstance(obj, dict) and "items" in obj else obj
        if not isinstance(items, list):
            raise ValueError("State extractor LLM output must be a JSON list or an object with an items list.")

        results: List[LocalState] = []
        for x in items:
            obligation = x.get("obligation", {})
            if not obligation and ("obligation.opens" in x or "obligation.resolves" in x):
                obligation = {
                    "opens": x.get("obligation.opens", []),
                    "resolves": x.get("obligation.resolves", []),
                }
            results.append(
                LocalState(
                    turn_id=int(x["turn_id"]),
                    speaker=str(x["speaker"]),
                    summary_topic=str(x["summary_topic"]),
                    intent=str(x["intent"]),
                    salient_entities=list(x.get("salient_entities", [])),
                    cue_markers=list(x.get("cue_markers", [])),
                    obligation=ObligationState(
                        opens=list(obligation.get("opens", [])),
                        resolves=list(obligation.get("resolves", [])),
                    ),
                )
            )

        expected_turn_ids = [u.turn_id for u in utterances]
        actual_turn_ids = [state.turn_id for state in results]
        if actual_turn_ids != expected_turn_ids:
            raise ValueError(
                "State extractor output turn_ids do not match the input dialogue order. "
                f"Expected {expected_turn_ids}, got {actual_turn_ids}."
            )
        return results
