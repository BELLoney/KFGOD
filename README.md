# KFGOD: Kernelized Fuzzy Approximation Fusion with Granular-ball Computing for Outlier Detection

This repository provides the official implementation of the following paper (Information Fusion, 2025, Accept):

> **A Kernelized Fuzzy Approximation Fusion Model with Granular-ball Computing for Outlier Detection**  
> Yongxiang Li, Xinyu Su, Zhong Yuan, Run Ye, Dezhong Peng, Hongmei Chen

## 📌 Introduction

Outlier detection is a key task in data analysis. This work proposes **KFGOD**, a method that addresses limitations of existing fuzzy rough set-based techniques by:

- Using **granular-ball computing** to capture both local and global outlier information with multi-granularity modeling.
- Applying **kernelized fuzzy rough sets** to model nonlinear relationships between samples.
- Performing **hierarchical information fusion** to estimate outlier degrees at both granular-ball and sample levels.

KFGOD is unsupervised, robust to noise, and effective on various datasets.

## 📁 Directory Structure

```
.
├── code/                                 # Source code
│   └── GB_generation_with_idx.py         # Granular-ball generation
│   └── KFGOD.py                          # Main entry point
├── datasets/                             # Benchmark datasets used in the paper
└── README.md                             # Project readme
```

## 🚀 How to Run

### 1. Install dependencies

Ensure Python 3.8+ is installed. Required packages include:

- numpy  
- scikit-learn  

Install with:

```bash
pip install numpy scikit-learn
```

### 2. Run the code

```bash
cd code
python KFGOD.py
```

You can modify dataset paths and parameters directly in `KFGOD.py`.
