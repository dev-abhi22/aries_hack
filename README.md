#  AI Customer Complaint Classification Ensemble

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

##  Project Overview
This repository contains a high-performance solution for the automated classification of customer complaints. The system is designed to analyze unstructured text data and predict three key attributes simultaneously:
1.  **Primary Category:** The main topic of the complaint (e.g.,"Credit Cards" ).
2.  **Secondary Category:** A more granular sub-topic (e.g., "Billing Dispute").
3.  **Severity Score:** A regression score (1-5) indicating the urgency of the complaint.

The solution utilizes a **Heterogeneous Ensemble** of two state-of-the-art Transformer models (**DeBERTa v3** and **RoBERTa**), trained using a multi-task learning approach and combined via weighted blending for maximum accuracy.

---

##  Methodology & Architecture

### 1. Dual-Model Architecture
To capture both complex semantic dependencies and simple keyword patterns, we employ two distinct transformer architectures:

* **Model A: Microsoft DeBERTa v3 Base**
    * *Why:* Superior at understanding disentangled attention and relative positions. It handles complex logic and sarcasm exceptionally well.
    * *Role:* The "Logic Expert" of the ensemble.
* **Model B: RoBERTa Base**
    * *Why:* Robustly optimized BERT variant. Excellent at spotting key phrases and maintaining stability.
    * *Role:* The "Safety Net" of the ensemble.

### 2. Multi-Task Learning Head
Instead of training three separate models for the three targets, we use a single shared encoder with three specific heads. This allows the model to learn shared representations (e.g., a "severe" complaint usually has specific "billing" keywords).



### 3. Training Strategy
* **Cross-Validation:** Stratified 5-Fold Cross-Validation (Training on 80%, Validating on 20%).
* **Epochs:** 10 Epochs per fold (ensuring deep convergence).
* **Loss Function:** Weighted Sum Loss:
    $$Loss = 0.3 \times Loss_{Primary} + 0.4 \times Loss_{Secondary} + 0.3 \times Loss_{Severity}$$

---

##  Inference & Ensemble Logic

We do not trust the models equally. Since DeBERTa v3 generally outperforms RoBERTa on NLU tasks, we implement a **Weighted Blending** strategy during inference to boost performance.

### The Blending Formula
$$FinalPrediction = (0.6 \times P_{DeBERTa}) + (0.4 \times P_{RoBERTa})$$

* **DeBERTa (60% Weight):** Acts as the primary decision-maker.
* **RoBERTa (40% Weight):** Provides a stabilizing vote, correcting DeBERTa when it is uncertain.

---

##  Repository Structure

```text
│
├── README.md                  #  Documentation (Updated text below)
├── model_train_ensemble.ipynb #  Your Training Notebook (10 Epochs)
├── final_classification.ipynb #  Your Inference Notebook (Weighted Blending)
└── submission.csv             #  The final output file
