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

OUTPUT_PATH = REPO_ROOT / "outputs" / "example_from_openai.json"


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    dialogue = load_sample_dialogue()
    pipeline = DialogueSegmentationPipeline.from_openai(
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    result = pipeline.run(dialogue, dialogue_id="dlg_sample")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved result to {OUTPUT_PATH}")
    print("Episodes:")
    for episode in result["episodes"]:
        print(f"  {episode['episode_id']} -> {episode['utterance_span']}")


if __name__ == "__main__":
    main()
