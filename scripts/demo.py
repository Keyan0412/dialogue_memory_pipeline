from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.pipeline import DialogueSegmentationPipeline, load_dialogue


DEFAULT_INPUT = ROOT / "src" / "dialogue_memory_pipeline" / "data" / "sample_dialogue.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "demo_output.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full dialogue memory pipeline demo.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input dialogue JSON. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3.5-plus"), help="Model name for the OpenAI-compatible API.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    dialogue = load_dialogue(args.input)
    config = PipelineConfig()
    pipeline = DialogueSegmentationPipeline.from_env(model=args.model, config=config)
    result = pipeline.run(dialogue)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[demo] Final segments:")
    for seg in result["segments"]:
        print(f"  {seg['segment_id']} | span={seg['utterance_span']} | topic={seg['segment_state']['stable_topic']}")

    print("\n[demo] Episodic memories:")
    for ep in result["episodes"]:
        print(f"  {ep['episode_id']} | span={ep['utterance_span']}")
        print(f"    summary: {ep['retrieval_summary']}")

    print(f"\n[demo] Saved full result to: {args.output}")


if __name__ == "__main__":
    main()
