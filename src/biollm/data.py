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


def kb_to_positive_pair_examples(ds: Dataset) -> List[PairExample]:
    """
    Convert BigBio KB rows into positive pair examples.

    Notes
    -----
    BigBio KB schema provides:
    - passages: list of texts
    - entities: list of entity objects with IDs and mentions
    - relations: list of typed relations referencing entity IDs

    We create examples by:
    - Picking the first passage text (MVP)
    - Mapping entity_id -> first surface form mention
    - Emitting (text, e1, e2, relation_type) for each annotated relation
    """
    examples: List[PairExample] = []

    for row in ds:
        passages = row.get("passages", [])
        if not passages:
            continue

        # MVP choice: take the first passage text
        # BigBio stores passage text as a list of strings; we take the first.
        text = passages[0]["text"][0] if passages[0].get("text") else ""
        if not text:
            continue

        # Build entity_id -> mention string
        ent_text: Dict[str, str] = {}
        for ent in row.get("entities", []):
            ent_id = ent.get("id", "")
            # ent["text"] often is a list of mentions; take the first
            mention = ent["text"][0] if ent.get("text") else ""
            if ent_id and mention:
                ent_text[ent_id] = mention

        # Create positive examples from labeled relations
        for rel in row.get("relations", []):
            arg1_id = rel.get("arg1_id", "")
            arg2_id = rel.get("arg2_id", "")
            rel_type = rel.get("type", "")

            e1 = ent_text.get(arg1_id, "")
            e2 = ent_text.get(arg2_id, "")
            if not (e1 and e2 and rel_type):
                continue

            examples.append(PairExample(text=text, e1=e1, e2=e2, label=rel_type))

    return examples
