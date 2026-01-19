"""
Inference utility for biomedical relation classification.

This version is designed for real-world usage where a NO_RELATION class exists.
It provides:
- Full label probability distribution
- Top-k predicted labels
- Best non-negative (non-NO_RELATION) label and score
"""

import argparse
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .constants import DEFAULT_BASE_MODEL, DEFAULT_MAX_LENGTH, SEP


@torch.inference_mode()
def predict_proba(
    model_name_or_path: str,
    sentence: str,
    e1: str,
    e2: str,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Predict probability distribution over relation labels.

    Parameters
    ----------
    model_name_or_path : str
        Hugging Face model name or local fine-tuned checkpoint path.
    sentence : str
        Input sentence/passage containing relation context.
    e1 : str
        Entity1 surface form (e.g., disease or drug).
    e2 : str
        Entity2 surface form (e.g., gene/protein).
    device : str
        "cpu", "cuda", or "mps" (Apple Silicon).

    Returns
    -------
    Dict[str, float]
        Mapping from label -> probability.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    # Prompt-like input format (consistent with training)
    text = f"Sentence: {sentence}{SEP}Entity1: {e1}{SEP}Entity2: {e2}"

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=DEFAULT_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: [num_labels]

    id2label = model.config.id2label
    return {id2label[i]: float(probs[i].detach().cpu()) for i in range(probs.shape[0])}


def top_k(scores: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
    """
    Return top-k labels by probability (descending).
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def best_non_negative(
    scores: Dict[str, float],
    negative_label: str = "NO_RELATION",
) -> Tuple[str, float]:
    """
    Return the best label excluding the negative label (NO_RELATION).

    If all labels are negative or only one label exists, returns (negative_label, score).
    """
    if negative_label not in scores:
        # If the model doesn't have NO_RELATION, just return best overall.
        best = max(scores.items(), key=lambda x: x[1])
        return best[0], best[1]

    filtered = [(k, v) for k, v in scores.items() if k != negative_label]
    if not filtered:
        return negative_label, float(scores.get(negative_label, 0.0))

    best = max(filtered, key=lambda x: x[1])
    return best[0], best[1]


def format_sorted(scores: Dict[str, float]) -> str:
    """
    Pretty-print all label probabilities in descending order.
    """
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lines = [f"{label:>25s}  {p:.4f}" for label, p in items]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relation classification inference.")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                        help="HF model name or fine-tuned checkpoint path.")
    parser.add_argument("--sentence", type=str, required=True, help="Input sentence/passage.")
    parser.add_argument("--e1", type=str, required=True, help="Entity1 string.")
    parser.add_argument("--e2", type=str, required=True, help="Entity2 string.")
    parser.add_argument("--device", type=str, default="cpu",
                        help='cpu | cuda | mps (Apple Silicon).')
    parser.add_argument("--topk", type=int, default=5, help="Show top-k labels.")
    parser.add_argument("--negative_label", type=str, default="NO_RELATION",
                        help="Name of the negative class.")
    parser.add_argument("--show_all", action="store_true",
                        help="If set, print all label probabilities.")
    args = parser.parse_args()

    scores = predict_proba(
        model_name_or_path=args.model,
        sentence=args.sentence,
        e1=args.e1,
        e2=args.e2,
        device=args.device,
    )

    # 1) Top-k overall (including NO_RELATION)
    print("\nTop predictions (including NO_RELATION):")
    for label, p in top_k(scores, k=args.topk):
        print(f"{label:>25s}  {p:.4f}")

    # 2) Best non-negative label (excluding NO_RELATION)
    best_label, best_score = best_non_negative(scores, negative_label=args.negative_label)
    print("\nBest non-negative prediction (excluding NO_RELATION):")
    print(f"{best_label:>25s}  {best_score:.4f}")

    # 3) Optional: show full distribution
    if args.show_all:
        print("\nAll label probabilities (sorted):")
        print(format_sorted(scores))


if __name__ == "__main__":
    main()
