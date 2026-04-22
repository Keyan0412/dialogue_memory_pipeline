from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.pipeline import DialogueSegmentationPipeline, load_dialogue


DEFAULT_INPUT = SRC / "dialogue_memory_pipeline" / "data" / "sample_dialogue.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "demo_output.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full dialogue memory pipeline demo.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input dialogue JSON. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3.6-flash"), help="Model name for the OpenAI-compatible API.")
    parser.add_argument("--local-state-chunk-size", type=int, default=0, help="Chunk size for local state extraction. 0 disables chunking.")
    parser.add_argument("--local-state-max-parallel", type=int, default=1, help="Max parallel local-state chunk requests.")
    parser.add_argument(
        "--local-state-transport",
        choices=["default", "bailian_batch_chat"],
        default="default",
        help="Transport for local state extraction. 'bailian_batch_chat' uses DashScope batch chat for chunk requests.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    dialogue = load_dialogue(args.input)
    config = PipelineConfig(
        local_state_chunk_size=args.local_state_chunk_size,
        local_state_max_parallel=args.local_state_max_parallel,
        local_state_transport=args.local_state_transport,
    )
    pipeline = DialogueSegmentationPipeline.from_env(model=args.model, config=config)
    result = pipeline.run(dialogue, dialogue_id=args.input.stem)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[demo] Final segments:")
    for seg in result["segments"]:
        print(f"  {seg['segment_id']} | span={seg['utterance_span']} | topic={seg['segment_state']['stable_topic']}")

    print("\n[demo] Episodic memories:")
    for ep in result["episodes"]:
        print(f"  {ep['episode_id']} | span={ep['utterance_span']}")
        print(f"    summary_zh: {ep['retrieval_summary_zh']}")
        print(f"    summary_en: {ep['retrieval_summary_en']}")

    print(f"\n[demo] Saved full result to: {args.output}")


if __name__ == "__main__":
    main()
