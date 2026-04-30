
import json
from pathlib import Path

input_dir = Path("data/raw_samples")
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

with open("data/labels/difficulty_level.json") as f:
    valid_difficulty = json.load(f)

with open("data/labels/error_types.json") as f:
    valid_errors = json.load(f)

for file in input_dir.glob("*.json"):
    with open(file, encoding="utf-8") as f:
        raw = json.load(f)

    if raw.get("difficulty_level") not in valid_difficulty:
        print(f"SKIP {file.name}: invalid difficulty_level")
        continue

    if raw.get("error_type") not in valid_errors:
        print(f"SKIP {file.name}: invalid error_type")
        continue

    cleaned = {
        "id": raw.get("id", ""),
        "language": raw.get("language", "unknown").lower().strip(),
        "difficulty_level": raw.get("difficulty_level"),
        "error_type": raw.get("error_type"),
        "code": raw.get("code", "").strip(),
        "error_message": raw.get("error_message", "").strip()
    }

    out_file = output_dir / file.name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Cleaned: {file.name}")

print("All done!")

