CANDIDATE_GENERATION_PROMPT = (
    "You score every possible dialogue segment boundary. "
    "Return only JSON with key items. "
    "items must be a list with exactly one object for each possible boundary between adjacent turns. "
    "Each object must contain: boundary_after_turn, score, reasoning. "
    "boundary_after_turn is the turn index after which a split may occur. "
    "score is a confidence from 0 to 1, where higher means more likely to be a true segment boundary. "
    "reasoning is a short explanation. "
    "You must provide one score for every boundary from 0 through the last valid boundary."
)
