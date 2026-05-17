"""
Assemble a HuggingFace Dataset from text prompts + rendered images and push to Hub.

Usage:
    HF_TOKEN=<your_token> python scripts/02_push_dataset.py --repo YOUR_HF_USERNAME/vision_scope_prompts
"""

import argparse
import pathlib
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

TOPICS = [
    "biology", "math", "physics", "chemistry", "history",
    "geography", "literature", "cs", "linguistics", "arts",
]


def build_records() -> list[dict]:
    records = []
    for lang, fname in [("en", "questions_en.txt"), ("fr", "questions_fr.txt")]:
        lines = (DATA_DIR / fname).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 100, f"Expected 100 lines in {fname}, got {len(lines)}"

        for idx, text in enumerate(lines):
            img_path = IMG_DIR / f"{idx:03d}_{lang}.png"
            assert img_path.exists(), f"Missing image: {img_path}. Run 01_create_images.py first."

            records.append({
                "id": f"{idx:03d}_{lang}",
                "language": lang,
                "topic": TOPICS[idx // 10],
                "text": text,
                "image": Image.open(img_path).convert("RGB"),
            })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="YOUR_HF_USERNAME/vision_scope_prompts",
        help="HuggingFace repo id to push to, e.g. myname/vision_scope_prompts",
    )
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    args = parser.parse_args()

    print("Building records...")
    records = build_records()

    features = Features({
        "id": Value("string"),
        "language": Value("string"),
        "topic": Value("string"),
        "text": Value("string"),
        "image": HFImage(),
    })

    ds = Dataset.from_list(records, features=features)
    print(f"Dataset: {ds}")

    print(f"Pushing to {args.repo} ...")
    ds.push_to_hub(args.repo, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
