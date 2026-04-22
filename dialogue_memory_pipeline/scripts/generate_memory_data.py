import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv()

from dialogue_memory_pipeline.config import PipelineConfig
from dialogue_memory_pipeline.pipeline import DialogueSegmentationPipeline
from dialogue_memory_pipeline.core.schemas import Utterance

def main() -> None:
    config = PipelineConfig()
    pipeline = DialogueSegmentationPipeline.from_env(model="qwen3.5-plus", config=config)

    # Load dialogues
    dialogue_path = "/home/jan/repos/dialogue_memory_pipeline/src/dialogue_memory_pipeline/data/synthetic_recall_benchmark_dialogues_v2.json"
    with open(dialogue_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogues = data["dialogues"]

    # Process dialogues and extract memories
    memories = []
    for dialogue in dialogues:
        utterances = [Utterance(turn_id=int(x["turn_id"]), speaker=x["speaker"], text=x["text"]) for x in dialogue["turns"]]
        result = pipeline.run(utterances)
        episode = pipeline.extract_memories(result)
        memories.extend(episode)

    # Save extracted memories
    output_path = ROOT / "outputs" / "extracted_memories.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
