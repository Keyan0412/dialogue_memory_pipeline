from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from dialogue_memory_pipeline.clients.llm_client import OpenAIJSONLLM
from dialogue_memory_pipeline.core.schemas import Utterance
from dialogue_memory_pipeline.modules.state_extractor import LLMStateExtractor

DEFAULT_INPUT = ROOT / "src" / "dialogue_memory_pipeline" / "data" / "sample_dialogue.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "state_extractor_llm_output.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-backed state extractor on a dialogue sample.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the input dialogue JSON. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the extracted local states. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
        help="Model name for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start turn index to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of turns to send to the extractor. Use 0 for all turns.",
    )
    return parser.parse_args()


def slice_utterances(utterances: list[Utterance], start: int, limit: int) -> list[Utterance]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if start >= len(utterances):
        raise ValueError(f"--start {start} is out of range for {len(utterances)} utterances")
    if limit == 0:
        return utterances[start:]
    return utterances[start : start + limit]


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

    # Process the dialogue
    utterances = load_dialogue(args.input)
    sample = slice_utterances(utterances, args.start, args.limit)

    # Run the extractor
    llm = OpenAIJSONLLM(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
    )
    extractor = LLMStateExtractor(llm)
    started_at = time.perf_counter()
    states = extractor.extract(sample)
    elapsed = time.perf_counter() - started_at

    payload = {
        "model": args.model,
        "input_path": str(args.input),
        "turn_ids": [u.turn_id for u in sample],
        "elapsed_seconds": elapsed,
        "states": [state.to_dict() for state in states],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed turns: {payload['turn_ids']}")
    print(f"LLM response time: {elapsed:.2f}s")
    print(f"Saved extractor output to: {args.output}")
    for utterance, state in zip(sample, states):
        print(f"\nTurn {state.turn_id} | {state.speaker}")
        print(f"  text: {utterance.text}")
        print(f"  topic: {state.summary_topic}")
        print(f"  intent: {state.intent}")
        print(f"  entities: {', '.join(state.salient_entities) if state.salient_entities else '-'}")
        print(f"  cues: {', '.join(state.cue_markers) if state.cue_markers else '-'}")
        print(f"  obligation.opens: {', '.join(state.obligation.opens) if state.obligation.opens else '-'}")
        print(f"  obligation.resolves: {', '.join(state.obligation.resolves) if state.obligation.resolves else '-'}")


if __name__ == "__main__":
    main()
