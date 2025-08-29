<img width="6000" height="1057" alt="Methodology" src="https://github.com/user-attachments/assets/3bb6e5cd-97d2-45fb-a3bd-5da5136e1e86" /># Hy-GeneNet: A Hybrid Ensemble for DNA Gene Type Classification


This repository contains the official source code and resources for the paper: **"Hy-GeneNet: A Hybrid Ensemble of N-Gram and Deep Learning Models for Reliable and Interpretable DNA Gene Type Classification"**.

## 📖 Table of Contents
* [Introduction](#-introduction)
* [Key Features](#-key-features)
* [Model Architecture](#-model-architecture)
* [Results](#-results)
* [System Requirements](#-system-requirements)
* [Installation](#-installation)
* [Usage](#-usage)
  * [1. Data Preprocessing](#1-data-preprocessing)
  * [2. Training the Models](#2-training-the-models)
  * [3. Running the Ensemble](#3-running-the-ensemble)
  * [4. Explainable AI (XAI) Analysis](#4-explainable-ai-xai-analysis)
* [How to Cite](#-how-to-cite)
* [License](#-license)
<img width="6000" height="1057" alt="Methodology" src="https://github.com/user-attachments/assets/bdbcd3e6-909b-41fd-9b8b-b8829ba17165" />

## 🌐 Introduction
Hy-GeneNet is a novel hybrid ensemble model designed for the accurate and interpretable classification of DNA gene types. The rapid growth of genomic data necessitates robust tools for analysis, and Hy-GeneNet addresses this by synergistically combining a traditional N-gram model with a modern attention-based deep learning architecture. This fusion allows the model to capture both local statistical patterns and long-range dependencies within DNA sequences, leading to state-of-the-art performance.

This repository provides all the necessary code to replicate our experiments, train the models on your own data, and explore the model's predictions using Explainable AI (XAI) techniques.

## ✨ Key Features
- **Hybrid Ensemble Approach:** Fuses a TF-IDF N-gram model (with Random Forest) and an Attention-based Bidirectional LSTM for enhanced accuracy.
- **High Performance:** Achieves **98.26% accuracy** and a perfect **1.00 ROC AUC** score on our test dataset.
- **Interpretable by Design:** Includes scripts for XAI analysis using SHAP (for the N-gram model) and Attention Visualization (for the LSTM), ensuring that the model's decisions are transparent and biologically plausible.
- **End-to-End Pipeline:** Provides a complete workflow from data preprocessing and model training to evaluation and interpretation.


## 🏗️ Model Architecture
Hy-GeneNet leverages a soft-voting ensemble strategy, averaging the prediction probabilities of two powerful base models:
1.  **N-gram + Random Forest:** Captures local sequence motifs and statistical features using 3-gram to 6-gram representations.
2.  **Attention-based Bi-LSTM:** Learns complex, long-range dependencies in the sequence data using a multi-head attention mechanism.

The final architecture is illustrated below:

<img width="7500" height="4062" alt="arc" src="https://github.com/user-attachments/assets/189492a2-9c47-4eea-98b5-fcd5a301a0c1" />


## 📊 Results
Our hybrid model significantly outperforms its individual components and establishes a new state-of-the-art benchmark on the evaluated dataset.

| Model               | Accuracy | Precision | Recall  | F1-Score | ROC AUC |
| ------------------- | :------: | :-------: | :-----: | :------: | :-----: |
| N-Gram (RF)         | 98.01%   | 97.98%    | 98.01%  | 97.95%   | 0.99    |
| Attention LSTM      | 97.85%   | 97.80%    | 97.85%  | 97.79%   | 1.00    |
| **Hy-GeneNet (Ensemble)** | **98.26%** | **98.24%**  | **98.26%**| **98.22%** | **1.00**  |

## 🖥️ System Requirements
- Python 3.8+
- PyTorch 1.10+
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- SHAP
- CUDA-enabled GPU (recommended for training the LSTM model)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Hy-GeneNet.git
    cd Hy-GeneNet
    ```
    <!--- IMPORTANT: Replace 'your-username' with your actual GitHub username --->

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage
The source code is organized into scripts for each stage of the pipeline.
python preprocess.py --data_path /path/to/your/data.csv --output_dir /path/to/processed_data
