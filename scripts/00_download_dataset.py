"""
Quick sanity check: download and inspect a biomedical relation dataset.

We use BigBio's ChemProt (knowledge-base schema) because it is:
- Public and well-known in biomedical relation extraction
- Naturally compatible with downstream knowledge graph construction
"""

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")
    print(ds)
    print("\nTrain features:\n", ds["train"].features)


if __name__ == "__main__":
    main()
