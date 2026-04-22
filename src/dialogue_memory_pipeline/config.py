from dataclasses import dataclass
from typing import Literal


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

    # ----------------------------
    # Local state extraction
    # ----------------------------
    local_state_chunk_size: int = 0
    local_state_max_parallel: int = 1
    local_state_transport: Literal["default", "bailian_batch_chat"] = "default"
