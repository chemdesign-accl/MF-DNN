# MF-DNN: Molecular Fingerprint–Deep Neural Network

This repository contains the implementation of **MF-DNN**, a deep learning model developed to aid in the discovery of novel **SARS-CoV-2 nsp13 helicase inhibitors**, including the identification of **MWAC-3429 hits**.  

MF-DNN was primarily used as a **pre-filtering model** in the drug discovery pipeline to prioritize compounds from large virtual screening (VS) libraries for downstream simulation and experimental validation.

---

## Overview

MF-DNN combines **Morgan fingerprints** (ECFP-like circular fingerprints) with a **fully connected feed-forward neural network** trained to discriminate between active and inactive compounds.  

The workflow:

1. Generate **1024-bit Morgan fingerprints** (radius 4) from SMILES strings.  
2. Train MF-DNN using active/inactive labeled compounds.  
3. Apply trained model to blind and evaluation sets.  
4. Output predictions for compound prioritization.  

<p align="center">
  <img src="./Fig_2.png" width="300" />
</p>

*Figure 1. MF-DNN architecture used in the study:[EDIT].


---

## Key Features

- **Input**: SMILES strings → Morgan fingerprints  
- **Architecture**:  
  - Input layer (1024 bits)  
  - Hidden layer (32 units, ReLU)  
  - Output layer (Sigmoid)  
- **Loss**: Binary Cross-Entropy  
- **Optimizer**: Adam (LR=0.0001, weight decay=0.001)  
- **Training**: 70 epochs, batch size = 8  
- **Metrics**: Accuracy, F1 score, ROC-AUC, Confusion matrix  

---

## Repository Contents

### Files for MF-DNN Predictor
- `main_Code.py` → Model implementation and training pipeline  
- `dataset.csv` → Training dataset   
- `for_blind_nluc.csv` → Blind test set  
- `unique_compounds.csv` → MWAC-2380, MWAC-2381, and MWAC-2384 virtual screening Enamine Libraries analogs  

### Files for Associated Content (Molecular Dynamics Simulations & Analysis)
- `Scripts_Input_Files_MDs.zip` → Input files required for molecular simulations, including:  
  - Parameters  
  - Starting poses  
  - Representative PDB coordinates  
  - Analysis scripts for clustering, contacts, and MM/GBSA calculations  


---

## Quickstart

### 1. Create Environment

```bash
conda create -n mfdnn python=3.9 -y
conda activate mfdnn
```
### 2. Install Dependencies
```bash
conda install -c conda-forge rdkit -y
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn

```
## Usage

### 1. Training & Blind Set Evaluation and Prediction

```bash
python main_code.py
