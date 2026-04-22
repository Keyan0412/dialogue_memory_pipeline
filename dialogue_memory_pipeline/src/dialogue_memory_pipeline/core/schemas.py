from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class Utterance:
    turn_id: int
    speaker: str
    text: str


@dataclass
class ObligationState:
    opens: List[str] = field(default_factory=list)
    resolves: List[str] = field(default_factory=list)


@dataclass
class LocalState:
    turn_id: int
    speaker: str
    summary_topic: str
    intent: str
    salient_entities: List[str] = field(default_factory=list)
    cue_markers: List[str] = field(default_factory=list)
    obligation: ObligationState = field(default_factory=ObligationState)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentState:
    stable_topic: str
    discourse_goal: str
    focus_topics: List[str] = field(default_factory=list)
    entity_core: List[str] = field(default_factory=list)
    open_obligations: List[str] = field(default_factory=list)
    dominant_relation: str = "continue"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateBoundary:
    boundary_after_turn: int
    score: float
    left_turn_id: Optional[int] = None
    right_turn_id: Optional[int] = None
    left_text: Optional[str] = None
    right_text: Optional[str] = None
    reasoning: str = ""
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransitionDecision:
    transition: str
    confidence: float
    should_split: bool
    reasoning: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    segment_id: str
    utterances: List[Utterance]
    local_states: List[LocalState]
    segment_state: SegmentState

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "utterance_span": [self.utterances[0].turn_id, self.utterances[-1].turn_id],
            "utterances": [asdict(u) for u in self.utterances],
            "local_states": [ls.to_dict() for ls in self.local_states],
            "segment_state": self.segment_state.to_dict(),
        }


@dataclass
class EpisodeMemory:
    episode_id: str
    utterance_span: List[int]
    utterances: List[Utterance]
    retrieval_summary_zh: str
    retrieval_summary_en: str
    key_entities_zh: List[str] = field(default_factory=list)
    key_entities_en: List[str] = field(default_factory=list)
    importance: int = 1

    @property
    def retrieval_summary(self) -> str:
        return self.retrieval_summary_en

    @property
    def key_entities(self) -> List[str]:
        return self.key_entities_en

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "utterance_span": self.utterance_span,
            "utterances": [asdict(u) for u in self.utterances],
            "retrieval_summary_zh": self.retrieval_summary_zh,
            "retrieval_summary_en": self.retrieval_summary_en,
            "key_entities_zh": self.key_entities_zh,
            "key_entities_en": self.key_entities_en,
            "importance": self.importance,
        }
