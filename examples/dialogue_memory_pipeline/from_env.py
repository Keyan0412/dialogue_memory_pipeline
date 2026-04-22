from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

from dialogue_memory_pipeline import DialogueSegmentationPipeline, load_sample_dialogue

OUTPUT_PATH = REPO_ROOT / "outputs" / "example_from_env.json"


def main() -> None:
    load_dotenv()

    dialogue = load_sample_dialogue()
    pipeline = DialogueSegmentationPipeline.from_env()
    result = pipeline.run(dialogue, dialogue_id="dlg_sample")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved result to {OUTPUT_PATH}")
    print("Models:")
    print(f"  candidate: {os.getenv('OPENAI_MODEL_CANDIDATE', os.getenv('OPENAI_MODEL', 'qwen3.5-plus'))}")
    print(f"  local_state: {os.getenv('OPENAI_MODEL_LOCAL_STATE', os.getenv('OPENAI_MODEL', 'qwen3.5-plus'))}")
    print(f"  transition: {os.getenv('OPENAI_MODEL_TRANSITION', os.getenv('OPENAI_MODEL', 'qwen3.5-plus'))}")
    print(f"  memory: {os.getenv('OPENAI_MODEL_MEMORY', os.getenv('OPENAI_MODEL', 'qwen3.5-plus'))}")
    print("Segments:")
    for seg in result["segments"]:
        print(f"  {seg['segment_id']} -> {seg['utterance_span']}")


if __name__ == "__main__":
    main()
