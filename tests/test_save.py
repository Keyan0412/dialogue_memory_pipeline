import json
from pathlib import Path

memories = []

ROOT = Path(__file__).resolve().parents[1]
output_path = ROOT / "outputs" / "extracted_memories.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(memories, f, ensure_ascii=False, indent=4)
