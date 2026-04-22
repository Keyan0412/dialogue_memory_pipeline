import json
from pathlib import Path

memories = []

REPO_ROOT = Path(__file__).resolve().parents[2]
output_path = REPO_ROOT / "outputs" / "extracted_memories.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(memories, f, ensure_ascii=False, indent=4)
