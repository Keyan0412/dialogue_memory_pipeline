from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # ----------------------------
    # Candidate generation
    # ----------------------------
    top_p_candidates: float = 0.30
    min_candidate_score: float = 0.20

    # ----------------------------
    # Transition / segment logic
    # ----------------------------
    right_preview_window: int = 3
    min_segment_len: int = 2
