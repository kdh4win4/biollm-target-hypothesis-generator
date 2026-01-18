#!/usr/bin/env bash
set -euo pipefail

# Replace --model with your fine-tuned checkpoint path after training.
# Example: outputs/pubmedbert-chemprot/checkpoint-XXXX
python -m src.biollm.infer \
  --model "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" \
  --sentence "EGFR inhibitors reduced tumor growth in lung cancer." \
  --e1 "EGFR inhibitors" \
  --e2 "EGFR"
