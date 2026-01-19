"""
Data loading + lightweight conversion from BigBio KB schema
to a pairwise relation classification dataset.

This module intentionally implements a minimal viable pipeline:
- Use a single passage text per document (MVP)
- Create positive examples from annotated relations
- (Optional) Add negative samples later if needed
"""

from dataclasses import dataclass
from typing import Dict, List

from datasets import Dataset, load_dataset


@dataclass
class PairExample:
    """
    A single training example for relation classification.

    Attributes
    ----------
    text : str
        The sentence/passage containing the relation context.
    e1 : str
        Surface form of entity 1 (chemical/drug or protein).
    e2 : str
        Surface form of entity 2.
    label : str
        Relation label string (e.g., "INHIBITOR", "ACTIVATOR", etc.)
    """
    text: str
    e1: str
    e2: str
    label: str


def load_chemprot_bigbio(split: str) -> Dataset:
    """
    Load ChemProt dataset in BigBio KB schema.

    Parameters
    ----------
    split : str
        One of {"train", "validation", "test"} depending on dataset config.

    Returns
    -------
    Dataset
        Hugging Face Dataset object.
    """
    return load_dataset("bigbio/chemprot", "chemprot_bigbio_kb", split=split)


"""
Data loading + conversion from BigBio KB schema to pairwise relation classification.

Now includes:
- Positive examples from annotated relations
- Negative examples (NO_RELATION) via within-document entity-pair sampling
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import random

from datasets import Dataset, load_dataset


@dataclass
class PairExample:
    """
    A single training example for relation classification.
    """
    text: str
    e1: str
    e2: str
    label: str


def load_chemprot_bigbio(split: str) -> Dataset:
    """
    Load ChemProt dataset in BigBio KB schema.
    """
    return load_dataset("bigbio/chemprot", "chemprot_bigbio_kb", split=split)


def kb_to_pair_examples_with_negatives(
    ds: Dataset,
    negative_label: str = "NO_RELATION",
    neg_pos_ratio: float = 1.0,
    seed: int = 42,
    max_entities_per_doc: int = 40,
) -> List[PairExample]:
    """
    Convert BigBio KB rows into pair examples with negative sampling.

    Strategy (MVP but realistic):
    - Build positives from annotated relations (arg1_id, arg2_id, type)
    - Sample negatives from all possible entity pairs *within the same document*
      that are NOT present as positives.
    - Keep dataset size reasonable with:
        - max_entities_per_doc cap
        - sampling only up to neg_pos_ratio * num_positives per document

    Parameters
    ----------
    ds : Dataset
        BigBio KB split dataset.
    negative_label : str
        Label name for negatives.
    neg_pos_ratio : float
        Number of negatives per positive (approx). 1.0 is a good baseline.
    seed : int
        RNG seed for reproducibility.
    max_entities_per_doc : int
        Cap number of entities considered per doc to avoid combinatorial explosion.

    Returns
    -------
    List[PairExample]
        Mixed list of positives + negatives.
    """
    rng = random.Random(seed)
    out: List[PairExample] = []

    for row in ds:
        passages = row.get("passages", [])
        if not passages:
            continue

        # MVP: use first passage text
        text = passages[0]["text"][0] if passages[0].get("text") else ""
        if not text:
            continue

        # Map entity_id -> surface form (first mention)
        ent_text: Dict[str, str] = {}
        ent_ids: List[str] = []
        for ent in row.get("entities", []):
            ent_id = ent.get("id", "")
            mention = ent["text"][0] if ent.get("text") else ""
            if ent_id and mention:
                ent_text[ent_id] = mention
                ent_ids.append(ent_id)

        if len(ent_ids) < 2:
            continue

        # Limit entities per doc for efficiency
        if len(ent_ids) > max_entities_per_doc:
            rng.shuffle(ent_ids)
            ent_ids = ent_ids[:max_entities_per_doc]

        # Collect positives and track positive pairs by entity_id
        pos_pairs: Set[Tuple[str, str]] = set()
        n_pos = 0

        for rel in row.get("relations", []):
            arg1_id = rel.get("arg1_id", "")
            arg2_id = rel.get("arg2_id", "")
            rel_type = rel.get("type", "")

            e1 = ent_text.get(arg1_id, "")
            e2 = ent_text.get(arg2_id, "")
            if not (arg1_id and arg2_id and e1 and e2 and rel_type):
                continue

            out.append(PairExample(text=text, e1=e1, e2=e2, label=rel_type))
            pos_pairs.add((arg1_id, arg2_id))
            n_pos += 1

        # If no positives in this doc, skip negatives (keeps training stable for MVP)
        if n_pos == 0:
            continue

        # Generate candidate negative pairs (ordered pairs)
        all_pairs: List[Tuple[str, str]] = []
        for i in range(len(ent_ids)):
            for j in range(len(ent_ids)):
                if i == j:
                    continue
                pair = (ent_ids[i], ent_ids[j])
                if pair in pos_pairs:
                    continue
                all_pairs.append(pair)

        if not all_pairs:
            continue

        # Sample negatives: up to neg_pos_ratio * positives for this document
        n_neg = int(round(neg_pos_ratio * n_pos))
        n_neg = min(n_neg, len(all_pairs))

        sampled = rng.sample(all_pairs, n_neg)
        for (a1, a2) in sampled:
            e1 = ent_text.get(a1, "")
            e2 = ent_text.get(a2, "")
            if not (e1 and e2):
                continue
            out.append(PairExample(text=text, e1=e1, e2=e2, label=negative_label))

    return out
