"""
Train a LoRA-fine-tuned biomedical relation extraction model
using ChemProt (BigBio) dataset.

This script:
- Builds (sentence, entity1, entity2, label) training examples
- Adds NO_RELATION negative sampling
- Fine-tunes PubMedBERT with LoRA (PEFT)
- Saves labels.json for correct inference label restoration
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from .constants import DEFAULT_MAX_LENGTH, SEP

# =========================
# Config
# =========================

BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
OUTPUT_DIR = "outputs/pubmedbert-chemprot"
NEGATIVE_LABEL = "NO_RELATION"
MAX_LENGTH = DEFAULT_MAX_LENGTH
SEED = 42


# =========================
# Data structure
# =========================

@dataclass
class Example:
    text: str
    label: str


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text(sentence: str, e1: str, e2: str) -> str:
    return f"Sentence: {sentence}{SEP}Entity1: {e1}{SEP}Entity2: {e2}"


# =========================
# Dataset processing
# =========================

def build_examples(split) -> List[Example]:
    examples: List[Example] = []

    for row in split:
        passages = row["passages"]
        entities = {e["id"]: " ".join(e["text"]) for e in row["entities"]}
        relations = row["relations"]

        # collect all sentences
        sentences = []
        for p in passages:
            sentences.extend(p["text"])

        # positive examples
        for r in relations:
            e1 = entities.get(r["arg1_id"])
            e2 = entities.get(r["arg2_id"])
            if not e1 or not e2:
                continue

            for sent in sentences:
                text = build_text(sent, e1, e2)
                examples.append(Example(text=text, label=r["type"]))

        # negative sampling: random entity pairs
        ent_ids = list(entities.keys())
        for _ in range(len(relations)):
            if len(ent_ids) < 2:
                break
            a, b = random.sample(ent_ids, 2)
            for sent in sentences:
                text = build_text(sent, entities[a], entities[b])
                examples.append(Example(text=text, label=NEGATIVE_LABEL))

    return examples


# =========================
# Tokenization
# =========================

def tokenize_fn(examples, tokenizer, label2id):
    texts = [ex.text for ex in examples]
    labels = [label2id[ex.label] for ex in examples]

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    enc["labels"] = labels
    return enc


# =========================
# Main
# =========================

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] Loading ChemProt dataset...")
    dataset = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb")

    print("[2] Building training examples...")
    train_examples = build_examples(dataset["train"])
    val_examples = build_examples(dataset["validation"])

    # =========================
    # Build label space
    # =========================
    labels = sorted(list({ex.label for ex in train_examples}))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # 🔥 SAVE LABELS FOR INFERENCE
    labels_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"[Saved] labels.json → {labels_path}")

    print(f"[3] Labels ({len(labels)}): {labels}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("[4] Tokenizing...")
    train_enc = tokenize_fn(train_examples, tokenizer, label2id)
    val_enc = tokenize_fn(val_examples, tokenizer, label2id)

    from datasets import Dataset

    train_dataset = Dataset.from_dict(train_enc)
    val_dataset = Dataset.from_dict(val_enc)

    print("[5] Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    print("[6] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # =========================
    # Training args
    # =========================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("[7] Training start...")
    trainer.train()

    print("[8] Saving final model...")
    trainer.save_model(OUTPUT_DIR)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Model + labels saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

