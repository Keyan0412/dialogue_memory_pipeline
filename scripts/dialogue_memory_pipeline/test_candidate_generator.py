from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from dialogue_memory_pipeline.clients.llm_client import OpenAIJSONLLM
from dialogue_memory_pipeline.core.schemas import Utterance
from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.modules.candidate_generator import CandidateBoundaryGenerator


DEFAULT_INPUT = SRC / "dialogue_memory_pipeline" / "data" / "sample_dialogue.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "candidate_generator_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-backed candidate boundary generator.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input dialogue JSON. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON report. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Model name for the OpenAI-compatible API.")
    parser.add_argument("--top-p", type=float, default=0.30, help="Maximum returned-candidate ratio, relative to available boundaries.")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum accepted candidate confidence.")
    return parser.parse_args()


def load_dialogue(path: Path) -> list[Utterance]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [Utterance(turn_id=int(x["turn_id"]), speaker=x["speaker"], text=x["text"]) for x in items]


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    config = PipelineConfig(
        top_p_candidates=args.top_p,
        min_candidate_score=args.min_score,
    )
    utterances = load_dialogue(args.input)
    llm = OpenAIJSONLLM(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
    )
    generator = CandidateBoundaryGenerator(llm, config)
    all_boundaries = generator.score_all_boundaries(utterances)
    candidates = [candidate for candidate in all_boundaries if candidate.score >= config.min_candidate_score]
    candidates.sort(key=lambda x: x.score, reverse=True)
    boundary_count = max(0, len(utterances) - 1)
    candidate_limit = 0 if boundary_count == 0 else max(1, math.ceil(boundary_count * config.top_p_candidates))
    candidates = candidates[: candidate_limit]
    candidates.sort(key=lambda x: x.boundary_after_turn)

    payload = {
        "model": args.model,
        "input_path": str(args.input),
        "config": {
            "top_p_candidates": config.top_p_candidates,
            "min_candidate_score": config.min_candidate_score,
        },
        "all_boundaries": [candidate.to_dict() for candidate in all_boundaries],
        "candidates": [candidate.to_dict() for candidate in candidates],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not all_boundaries:
        print("No candidate boundaries returned.")
    else:
        for candidate in all_boundaries:
            print(f"{candidate.boundary_after_turn}->{candidate.right_turn_id}: {candidate.score:.3f}")


if __name__ == "__main__":
    main()
