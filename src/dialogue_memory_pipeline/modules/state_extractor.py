from __future__ import annotations

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from dialogue_memory_pipeline.clients.llm_client import JSONLLM
from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.core.schemas import LocalState, ObligationState, Utterance
from dialogue_memory_pipeline.prompts import CHINESE_EXTRACTION_SYSTEM_PROMPT


class BaseLocalStateExtractor(ABC):
    @abstractmethod
    def extract(self, utterances: Sequence[Utterance]) -> List[LocalState]:
        raise NotImplementedError


class LLMStateExtractor(BaseLocalStateExtractor):
    def __init__(self, llm: JSONLLM, config: PipelineConfig | None = None) -> None:
        self.llm = llm
        self.config = config or PipelineConfig()

    def extract(self, utterances: Sequence[Utterance]) -> List[LocalState]:
        if not utterances:
            return []

        chunk_size = max(0, int(self.config.local_state_chunk_size))
        if chunk_size <= 0 or len(utterances) <= chunk_size:
            return self._extract_chunk(utterances)

        chunks = [
            list(utterances[start : start + chunk_size])
            for start in range(0, len(utterances), chunk_size)
        ]
        max_parallel = max(1, int(self.config.local_state_max_parallel))
        if max_parallel == 1 or len(chunks) == 1:
            results = [self._extract_chunk(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                results = list(executor.map(self._extract_chunk, chunks))
        return [state for chunk in results for state in chunk]

    def _extract_chunk(self, utterances: Sequence[Utterance]) -> List[LocalState]:
        system_prompt = CHINESE_EXTRACTION_SYSTEM_PROMPT
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

        results = [self._parse_state(x) for x in items]
        self._validate_turn_ids(utterances, results)
        return results

    def _parse_state(self, x: dict) -> LocalState:
        obligation = x.get("obligation", {})
        if not obligation and ("obligation.opens" in x or "obligation.resolves" in x):
            obligation = {
                "opens": x.get("obligation.opens", []),
                "resolves": x.get("obligation.resolves", []),
            }
        return LocalState(
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

    def _validate_turn_ids(self, utterances: Sequence[Utterance], results: Sequence[LocalState]) -> None:
        expected_turn_ids = [u.turn_id for u in utterances]
        actual_turn_ids = [state.turn_id for state in results]
        if actual_turn_ids != expected_turn_ids:
            raise ValueError(
                "State extractor output turn_ids do not match the input dialogue order. "
                f"Expected {expected_turn_ids}, got {actual_turn_ids}."
            )


def compact_local_state_payload(state: LocalState) -> dict:
    """Return only the local-state fields that matter for downstream reasoning."""
    return {
        "turn_id": state.turn_id,
        "speaker": state.speaker,
        "summary_topic": state.summary_topic,
        "intent": state.intent,
        "obligation": {
            "opens": list(state.obligation.opens),
            "resolves": list(state.obligation.resolves),
        },
    }
