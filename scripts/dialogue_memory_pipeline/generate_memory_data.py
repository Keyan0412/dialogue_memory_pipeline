import os
import json
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv()

from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.pipeline import DialogueSegmentationPipeline
from dialogue_memory_pipeline.core.schemas import Utterance

DEFAULT_INPUT = SRC / "dialogue_memory_pipeline" / "data" / "synthetic_recall_benchmark_dialogues_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "episodes.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized episode exports from one or more dialogues.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input dialogue JSON. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--dialogue-id-prefix", default="dlg", help="Prefix used when an input dialogue has no dialogue_id.")
    return parser.parse_args()


def load_dialogue_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [{"dialogue_id": path.stem, "turns": data}]
    if isinstance(data, dict) and isinstance(data.get("dialogues"), list):
        return list(data["dialogues"])
    raise ValueError("Input JSON must be either a list of turns or an object with a 'dialogues' list.")


def main() -> None:
    config = PipelineConfig()
    pipeline = DialogueSegmentationPipeline.from_env(model="qwen3.5-plus", config=config)
    args = parse_args()

    records = []
    for index, dialogue in enumerate(load_dialogue_items(args.input)):
        dialogue_id = str(dialogue.get("dialogue_id") or f"{args.dialogue_id_prefix}_{index:04d}")
        utterances = [Utterance(turn_id=int(x["turn_id"]), speaker=x["speaker"], text=x["text"]) for x in dialogue["turns"]]
        result = pipeline.run(utterances, dialogue_id=dialogue_id)
        records.extend(pipeline.normalize_episode_records(result, dialogue_id=dialogue_id))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    print(f"Exported {len(records)} episode records to: {args.output}")

if __name__ == "__main__":
    main()
