"""
Inference helper: predict a relation label for a given sentence + entity pair.

This is the demo entry point to show hiring managers:
- You can fine-tune a biomedical transformer
- You can package a clean inference workflow
- Outputs can be converted into knowledge graph edges
"""

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .constants import DEFAULT_BASE_MODEL, DEFAULT_MAX_LENGTH, SEP


@torch.inference_mode()
def predict(
    model_name_or_path: str,
    sentence: str,
    e1: str,
    e2: str,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Return predicted label probabilities for the given entity pair context.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    text = f"Sentence: {sentence}{SEP}Entity1: {e1}{SEP}Entity2: {e2}"
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=DEFAULT_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

    id2label = model.config.id2label
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--e1", type=str, required=True)
    parser.add_argument("--e2", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    scores = predict(
        model_name_or_path=args.model,
        sentence=args.sentence,
        e1=args.e1,
        e2=args.e2,
        device=args.device,
    )

    # Print top-5 labels
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop predictions:")
    for label, p in top:
        print(f"{label:>20s}  {p:.4f}")


if __name__ == "__main__":
    main()
