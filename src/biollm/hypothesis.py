"""
Generate ranked target hypotheses from multiple sentences
using the fine-tuned relation classifier.

This simulates how LLM outputs feed target discovery.
"""

from collections import defaultdict
from typing import List, Dict, Tuple

from .infer import predict


def rank_targets(
    model_path: str,
    disease: str,
    sentences: List[str],
    candidate_targets: List[str],
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Rank targets by aggregated confidence scores.

    Parameters
    ----------
    model_path : str
        Path to fine-tuned model.
    disease : str
        Disease name (used as entity1).
    sentences : list of str
        Literature-derived sentences mentioning the disease.
    candidate_targets : list of str
        Genes / proteins to score.
    threshold : float
        Minimum probability to count as evidence.

    Returns
    -------
    List of (target, score) sorted descending.
    """
    scores = defaultdict(float)

    for sent in sentences:
        for target in candidate_targets:
            probs = predict(
                model_name_or_path=model_path,
                sentence=sent,
                e1=disease,
                e2=target,
            )

            # take max non-NO_RELATION score
            best = max(probs.values())
            if best >= threshold:
                scores[target] += best

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
