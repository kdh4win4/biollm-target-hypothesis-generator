#!/usr/bin/env bash
set -euo pipefail

# Train with LoRA (recommended baseline)
python -m src.biollm.train
