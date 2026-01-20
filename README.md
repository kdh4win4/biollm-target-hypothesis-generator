
---

# 🧠 BioLLM: Biomedical Relation Extraction for Target Discovery

**LLM-based hypothesis generation for AI-driven drug discovery**

---

## 🔬 Overview

This repository demonstrates an **end-to-end biomedical LLM pipeline for extracting biological relationships from literature** and transforming them into **structured candidate targets** for downstream analysis.

It is built as a portfolio project for **AI-driven target discovery and translational research**, following real-world industry patterns used in computational biology and drug discovery.

This project serves as the **upstream intelligence layer** of a larger discovery system:

LLM → biological relation extraction → candidate targets
↓
GNN → link prediction → prioritization (BioGNN / Repo2)

---

## 🚀 What This Project Demonstrates

### ✅ Technical Skills

* HuggingFace Transformers (PubMedBERT / BioBERT)
* Fine-tuning LLMs for relation classification
* Entity-aware biomedical text preprocessing
* Negative sampling with NO_RELATION class
* Reproducible CLI-based inference pipeline
* Integration-ready outputs for graph AI

### ✅ Drug Discovery Relevance

* Automated target discovery from literature
* Mechanism-of-action signal extraction
* Drug repurposing hypothesis generation
* Structured edge generation for knowledge graphs
* LLM → GNN pipeline integration

---

## 🧠 Model Objective

Given a sentence and two biological entities, the model predicts their biological relationship.

Example:

Sentence:
EGFR inhibitors reduced tumor growth in lung cancer.

Entity 1: lung cancer
Entity 2: EGFR

Model output:

* Downregulator (0.34)

This prediction becomes a **candidate edge** in a biomedical knowledge graph, later consumed by GNN models for prioritization.

---

## 🧬 Relation Types (ChemProt-style)

* Upregulator
* Downregulator
* Regulator
* Agonist
* Antagonist
* Substrate
* Inhibitor
* NO_RELATION (negative sampling)

---

## ⚙️ Pipeline

1. Load biomedical relation dataset (ChemProt via BigBio)
2. Fine-tune PubMedBERT for relation classification
3. Perform inference on new sentences
4. Output scored relations for graph construction
5. Pass relations to GNN-based prioritization (Repo2)

---

## 📂 Project Structure

biollm-target-hypothesis-generator/
├── src/biollm/
│   ├── data.py         # dataset preparation
│   ├── train.py        # LLM fine-tuning
│   └── infer.py        # relation inference
├── scripts/
│   ├── 00_download_dataset.py
│   ├── 02_train.sh
│   └── 03_demo.sh
├── outputs/
│   └── pubmedbert-chemprot/
└── README.md

---

## ▶️ How to Run

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train

```bash
bash scripts/02_train.sh
```

### 3) Inference

```bash
python -m src.biollm.infer \
  --model outputs/pubmedbert-chemprot \
  --sentence "EGFR inhibitors reduced tumor growth in lung cancer." \
  --e1 "lung cancer" \
  --e2 "EGFR"
```

---

## 🔍 Interpretation

* Output probabilities represent **biological evidence strength**
* High-confidence relations become **knowledge graph edges**
* These edges are consumed by GNNs for target prioritization
* Enables scalable hypothesis generation from literature

---

## 🔄 Integration with BioGNN (Repo2)

This project is designed to integrate directly with:

BioGNN: Drug–Target Interaction Link Prediction

Pipeline:
LLM (BioLLM) → relations
↓
Graph construction
↓
GNN (BioGNN) → ranked targets

Together, the two repositories form a **complete AI-driven target discovery pipeline**.

---

## 👤 Author

**Dohoon Kim**
Senior Computational Biologist / Data Scientist
Focus: AI for drug discovery, target identification, and translational biology

---

## ⭐ Why This Matters

This repository demonstrates:

* Real biomedical LLM fine-tuning (not prompt-only demos)
* Structured hypothesis generation for discovery science
* Integration-ready outputs for graph AI
* Practical ML engineering for drug discovery pipelines

These are **core skills required for AI Computational Biologist roles** in industry.

---
