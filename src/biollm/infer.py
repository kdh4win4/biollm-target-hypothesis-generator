"""
Inference utility for LoRA-fine-tuned biomedical relation classifier
with restored label names.
"""

import argparse
import os
import json
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig, PeftModel

from .constants import DEFAULT_MAX_LENGTH, SEP


def _infer_num_labels_from_adapter(model_path: str) -> int:
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
    sd = load_file(adapter_path)

    # key confirmed from your checkpoint
    return int(sd["base_model.model.classifier.weight"].shape[0])


@torch.inference_mode()
def predict_proba(
    model_path: str,
    sentence: str,
    e1: str,
    e2: str,
    device: str = "cpu",
) -> Dict[str, float]:

    # base model name
    peft_cfg = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # num_labels from adapter
    num_labels = _infer_num_labels_from_adapter(model_path)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()

    # load labels.json if exists
    labels_json = os.path.join(model_path, "labels.json")
    id2label = None
    if os.path.exists(labels_json):
        with open(labels_json, "r") as f:
            labels = json.load(f)
        id2label = {i: lab for i, lab in enumerate(labels)}

    text = f"Sentence: {sentence}{SEP}Entity1: {e1}{SEP}Entity2: {e2}"
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=DEFAULT_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    probs = torch.softmax(model(**enc).logits, dim=-1).squeeze(0)

    scores = {}
    for i in range(probs.shape[0]):
        label = id2label[i] if id2label else f"LABEL_{i}"
        scores[label] = float(probs[i].cpu())

    return scores


def top_k(scores: Dict[str, float], k: int = 5):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def best_non_negative(scores: Dict[str, float], negative_label="NO_RELATION"):
    items = [(k, v) for k, v in scores.items() if k != negative_label]
    return max(items, key=lambda x: x[1]) if items else max(scores.items(), key=lambda x: x[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sentence", required=True)
    p.add_argument("--e1", required=True)
    p.add_argument("--e2", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--topk", type=int, default=7)
    p.add_argument("--show_all", action="store_true")
    args = p.parse_args()

    scores = predict_proba(args.model, args.sentence, args.e1, args.e2, args.device)

    print("\nTop predictions:")
    for k, v in top_k(scores, args.topk):
        print(f"{k:>25s}  {v:.4f}")

    k, v = best_non_negative(scores)
    print("\nBest non-negative:")
    print(f"{k:>25s}  {v:.4f}")

    if args.show_all:
        print("\nAll:")
        for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{k:>25s}  {v:.4f}")


if __name__ == "__main__":
    main()

