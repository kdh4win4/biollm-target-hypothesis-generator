"""
Training script for biomedical relation classification using PubMedBERT.

This repo is intentionally positioned as:
- A practical LLM-adjacent workflow (transformers + fine-tuning)
- A scalable module that can feed edges into a biomedical knowledge graph
- A stepping stone toward later GNN / KG integration

We provide:
- Full fine-tuning option (baseline)
- Parameter-efficient LoRA fine-tuning option (recommended for laptops)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model

from .constants import DEFAULT_BASE_MODEL, DEFAULT_MAX_LENGTH, SEP
from .data import kb_to_pair_examples_with_negatives, load_chemprot_bigbio


def _build_hf_dataset(examples, label2id: Dict[str, int]) -> Dataset:
    """
    Convert list of PairExample objects into a HuggingFace Dataset.
    """
    return Dataset.from_dict(
        {
            "text": [ex.text for ex in examples],
            "e1": [ex.e1 for ex in examples],
            "e2": [ex.e2 for ex in examples],
            "label": [label2id[ex.label] for ex in examples],
        }
    )


def _tokenize_batch(tokenizer, batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
    """
    Build a prompt-like input for relation classification.

    Example format:
      Sentence: <text> [SEP] Entity1: <e1> [SEP] Entity2: <e2>

    This structure makes it easier to extend later to instruction-style models.
    """
    inputs = [
        f"Sentence: {t}{SEP}Entity1: {e1}{SEP}Entity2: {e2}"
        for t, e1, e2 in zip(batch["text"], batch["e1"], batch["e2"])
    ]
    return tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=DEFAULT_MAX_LENGTH,
    )


def _compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute macro-averaged classification metrics.
    Macro metrics are useful when class distribution is imbalanced.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main(
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "outputs/pubmedbert-chemprot",
    use_lora: bool = True,
    num_train_epochs: int = 3,
    lr: float = 2e-5,
) -> None:
    """
    Train a relation classifier on ChemProt (BigBio KB schema).

    Parameters
    ----------
    base_model : str
        Hugging Face model name (PubMedBERT recommended).
    output_dir : str
        Directory to save checkpoints and metrics.
    use_lora : bool
        If True, apply LoRA for parameter-efficient fine-tuning.
    num_train_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    """
    # Load dataset splits
    train_raw = load_chemprot_bigbio("train")
    test_raw = load_chemprot_bigbio("test")

    # Convert KB schema -> positive pair examples (MVP)
    train_examples = kb_to_pair_examples_with_negatives(
        train_raw,
        negative_label="NO_RELATION",
        neg_pos_ratio=1.0,   # start with 1:1 negatives:positives
        seed=42,
    )

    test_examples = kb_to_pair_examples_with_negatives(
        test_raw,
        negative_label="NO_RELATION",
        neg_pos_ratio=1.0,   # keep evaluation consistent
        seed=42,
    )


    # Build label space from training data
    labels = sorted(list({ex.label for ex in train_examples}))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Convert to HF datasets
    train_ds = _build_hf_dataset(train_examples, label2id)
    test_ds = _build_hf_dataset(test_examples, label2id)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Tokenization
    train_ds = train_ds.map(lambda b: _tokenize_batch(tokenizer, b), batched=True)
    test_ds = test_ds.map(lambda b: _tokenize_batch(tokenizer, b), batched=True)

    # Set PyTorch format
    columns = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)

    # Load classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Optional LoRA fine-tuning (recommended for limited GPU/CPU)
    if use_lora:
        # Note: target_modules may vary by architecture.
        # For BERT-like models, "query" and "value" are common targets.
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),  # enables fp16 when GPU supports it
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal eval metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
