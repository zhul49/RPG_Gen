import json
import os
import random
import sys
from pathlib import Path

# add the repo root to the path so the src imports below work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_prep import build_crd3, build_custom, build_fireball, build_light  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "processed" / "full"
VAL_FRAC = 0.05   # fraction of the final mix held out for validation
SEED = 42


# shuffle everything together then split off the validation set and write both files
def finalize(parts):
    all_examples = []
    for label, exs in parts:
        all_examples.extend(exs)
    random.shuffle(all_examples)
    n_val = max(1, round(len(all_examples) * VAL_FRAC))
    val = all_examples[:n_val]
    train = all_examples[n_val:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    with train_path.open("w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with val_path.open("w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(train)} train -> {train_path}")
    print(f"wrote {len(val)} val   -> {val_path}")
    src_counts = {}
    for ex in all_examples:
        src = ex.get("source", "?")
        src_counts[src] = src_counts.get(src, 0) + 1
    print("source breakdown:")
    for src in sorted(src_counts, key=src_counts.get, reverse=True):
        print(f"  {src}: {src_counts[src]}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment")

    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    custom = build_custom.build()
    crd3_kept, crd3_proactive_kept = build_crd3.build()
    light_examples = build_light.build()
    fireball_examples = build_fireball.build()

    parts = [
        ("crd3", crd3_kept),
        ("crd3_proactive", crd3_proactive_kept),
        ("light", light_examples),
        ("fireball", fireball_examples),
        ("custom", custom),
    ]
    finalize(parts)


if __name__ == "__main__":
    main()
