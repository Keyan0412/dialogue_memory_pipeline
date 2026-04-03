from __future__ import annotations

from collections import Counter
from typing import List, Sequence

from dialogue_memory_pipeline.core.schemas import LocalState, SegmentState, Utterance


RELATION_PRIORS = {
    "ask": "introduce",
    "answer": "respond",
    "respond": "respond",
    "elaborate": "elaborate",
    "reflect": "continue",
    "change_topic": "introduce",
}


def aggregate_segment_state(utterances: Sequence[Utterance], local_states: Sequence[LocalState]) -> SegmentState:
    if not utterances or not local_states:
        raise ValueError("Cannot aggregate an empty segment.")

    entity_counter = Counter(e.lower() for ls in local_states for e in ls.salient_entities)
    entities = [e for e, _ in entity_counter.most_common(6)]

    topic_tokens = Counter(tok for ls in local_states for tok in ls.summary_topic.split(" / "))
    stable_parts = [t for t, _ in topic_tokens.most_common(4)]
    stable_topic = " / ".join(stable_parts[:3]) if stable_parts else local_states[0].summary_topic

    focus_topics = [ls.summary_topic for ls in local_states[-3:]]

    open_obligations: List[str] = []
    resolved = set(x for ls in local_states for x in ls.obligation.resolves)
    for ls in local_states:
        for item in ls.obligation.opens:
            if item not in resolved:
                open_obligations.append(item)

    intent_counter = Counter(RELATION_PRIORS.get(ls.intent, "continue") for ls in local_states)
    dominant_relation = intent_counter.most_common(1)[0][0]

    discourse_goal = _infer_discourse_goal(local_states, stable_topic)

    return SegmentState(
        stable_topic=stable_topic,
        discourse_goal=discourse_goal,
        focus_topics=focus_topics,
        entity_core=entities,
        open_obligations=open_obligations,
        dominant_relation=dominant_relation,
    )



def _infer_discourse_goal(local_states: Sequence[LocalState], stable_topic: str) -> str:
    intents = Counter(ls.intent for ls in local_states)
    if intents.get("ask", 0) >= intents.get("answer", 0) and intents.get("ask", 0) >= 2:
        return f"Clarify or request guidance about {stable_topic}"
    if intents.get("answer", 0) + intents.get("elaborate", 0) >= max(2, len(local_states) // 2):
        return f"Explain or solve issues related to {stable_topic}"
    if intents.get("change_topic", 0) > 0:
        return f"Transition discussion around {stable_topic}"
    return f"Develop discussion around {stable_topic}"
