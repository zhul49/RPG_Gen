import subprocess
from pathlib import Path

from datasets import load_dataset, load_from_disk


RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def clone(url, dest):
    if dest.exists():
        print(f"  already at {dest.name}, skipping")
        return
    subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    print("LIGHT")
    light = RAW / "light"
    if not any(light.glob("**/*.arrow")):
        load_dataset("dap-exp/light_dialog").save_to_disk(str(light))
    for split, d in load_from_disk(str(light)).items():
        print(f"  {split}: {len(d):,} rows")

    print("\nCRD3")
    crd3 = RAW / "crd3_source"
    clone("https://github.com/RevanthRameshkumar/CRD3.git", crd3)
    eps = sorted((crd3 / "data" / "cleaned data").glob("*.json"))
    print(f"  {len(eps)} episode files")

    print("\nFIREBALL (sample only)")
    fb = RAW / "fireball_source"
    clone("https://github.com/zhudotexe/FIREBALL.git", fb)
    n = sum(1 for _ in (fb / "filtered_triples.jsonl").open())
    print(f"  filtered_triples.jsonl: {n} triples")


if __name__ == "__main__":
    main()
