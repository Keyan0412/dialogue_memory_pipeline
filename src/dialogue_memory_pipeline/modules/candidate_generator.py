from __future__ import annotations

import math
from typing import Any, List, Sequence

from dialogue_memory_pipeline.clients.llm_client import JSONLLM
from dialogue_memory_pipeline.core.schemas import CandidateBoundary, Utterance
from dialogue_memory_pipeline.prompts import CANDIDATE_GENERATION_PROMPT


class CandidateBoundaryGenerator:
    def __init__(self, llm: JSONLLM, config) -> None:
        self.llm = llm
        self.config = config

    def score_all_boundaries(self, utterances: Sequence[Utterance]) -> List[CandidateBoundary]:
        if len(utterances) < 2:
            return []

        system_prompt = CANDIDATE_GENERATION_PROMPT
        user_prompt = self._build_user_prompt(utterances)
        obj = self.llm.complete_json(system_prompt, user_prompt)
        items = obj.get("items", obj) if isinstance(obj, dict) else obj
        if not isinstance(items, list):
            raise ValueError("Candidate generator LLM output must be a JSON list or an object with an items list.")

        parsed: dict[int, CandidateBoundary] = {}
        for item in items:
            boundary = self._parse_candidate(item, utterances)
            if boundary is None:
                continue
            parsed[boundary.boundary_after_turn] = boundary

        all_boundaries: List[CandidateBoundary] = []
        for boundary_after_turn in range(len(utterances) - 1):
            boundary = parsed.get(boundary_after_turn)
            if boundary is None:
                boundary = CandidateBoundary(
                    boundary_after_turn=boundary_after_turn,
                    score=0.0,
                    left_turn_id=utterances[boundary_after_turn].turn_id,
                    right_turn_id=utterances[boundary_after_turn + 1].turn_id,
                    left_text=utterances[boundary_after_turn].text,
                    right_text=utterances[boundary_after_turn + 1].text,
                    reasoning="",
                    source="llm",
                )
            all_boundaries.append(boundary)
        return all_boundaries

    def generate(self, utterances: Sequence[Utterance]) -> List[CandidateBoundary]:
        candidates = self.score_all_boundaries(utterances)
        candidates = [x for x in candidates if x.score >= self.config.min_candidate_score]
        candidates.sort(key=lambda x: x.score, reverse=True)
        candidates = candidates[: self._candidate_limit(utterances)]
        candidates.sort(key=lambda x: x.boundary_after_turn)
        return candidates

    def _build_user_prompt(self, utterances: Sequence[Utterance]) -> str:
        candidate_limit = self._candidate_limit(utterances)
        lines = [
            f"Select at most {candidate_limit} candidate boundaries.",
            f"Ignore candidates with confidence below {self.config.min_candidate_score}.",
            "Dialogue:",
        ]
        lines.extend(f"[{u.turn_id}] {u.speaker}: {u.text}" for u in utterances)
        return "\n".join(lines)

    def _candidate_limit(self, utterances: Sequence[Utterance]) -> int:
        boundary_count = max(0, len(utterances) - 1)
        if boundary_count == 0:
            return 0
        top_p = min(1.0, max(0.0, float(self.config.top_p_candidates)))
        return max(1, math.ceil(boundary_count * top_p))

    def _parse_candidate(self, item: Any, utterances: Sequence[Utterance]) -> CandidateBoundary | None:
        if not isinstance(item, dict):
            return None

        boundary_value = item.get("boundary_after_turn", item.get("turn_id"))
        if boundary_value is None:
            return None
        boundary_after_turn = int(boundary_value)
        if boundary_after_turn < 0 or boundary_after_turn >= len(utterances) - 1:
            return None

        score = max(0.0, min(1.0, float(item.get("score", item.get("confidence", 0.0)))))
        reasoning = str(item.get("reasoning", "")).strip()

        return CandidateBoundary(
            boundary_after_turn=boundary_after_turn,
            score=score,
            left_turn_id=utterances[boundary_after_turn].turn_id,
            right_turn_id=utterances[boundary_after_turn + 1].turn_id,
            left_text=utterances[boundary_after_turn].text,
            right_text=utterances[boundary_after_turn + 1].text,
            reasoning=reasoning,
            source="llm",
        )
