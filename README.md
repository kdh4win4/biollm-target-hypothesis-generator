# BioLLM Target Hypothesis Generator

This repository demonstrates a practical LLM-adjacent workflow for **biomedical target discovery**:
fine-tuning a transformer model (PubMedBERT) to extract **biological relationships** from literature-derived text,
which can be integrated into downstream **knowledge graphs** and **target prioritization** pipelines.

## Why this matters
In translational discovery, key mechanistic evidence is scattered across literature and databases.
Automated relation extraction accelerates:
- target discovery and mechanism-of-action hypothesis generation
- drug repurposing signal collection
- building structured edges for biomedical knowledge graphs

## What this repo does
- Loads a public biomedical relation dataset (ChemProt via BigBio)
- Fine-tunes PubMedBERT for **relation classification**
- Provides a simple inference CLI to score candidate relations

### Target hypothesis demo
This repo can generate a ranked target list from disease-related sentences,
simulating AI-driven target discovery.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_download_dataset.py
bash scripts/02_train.sh
bash scripts/03_demo.sh
