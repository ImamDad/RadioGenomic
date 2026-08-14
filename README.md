# RadioGenomic
# MS-HGNN: Multi-Scale Hierarchical Graph Neural Network for NSCLC Prognosis

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

## 📝 Overview

MS-HGNN is a novel deep learning framework for concurrent **survival prediction** and **recurrence classification** in Non-Small Cell Lung Cancer (NSCLC). It integrates multimodal radiogenomic data (CT, PET, clinical, genomic) through a three-level hierarchical fusion strategy with biological grounding and uncertainty quantification.

### Key Features

- **🔬 Three-Level Hierarchical Fusion**: Cross-modal attention, biologically-informed heterogeneous graphs with semantic attention, and uncertainty-aware late fusion
- **🧬 Biological Grounding**: Three biologically-informed meta-paths (immune, proliferation, treatment response) with semantic attention validated against immunohistochemical biomarkers
- **📊 State-of-the-Art Performance**: C-index 0.85 for survival prediction, AUC 0.89 for recurrence classification
- **⚡ Model Compression**: 50% pruning + INT8 quantization reduces parameters from 28.5M to 14.2M with 28ms inference
- **🔍 Interpretability**: Multi-faceted visualization including feature importance, attention heatmaps, and patient-specific graphs
- **🌐 External Validation**: Validated on TCGA-LUAD (C-index 0.81 after fine-tuning)

## 📖 Paper

Dad, I., & He, J. (2026). *MS-HGNN: Interpretable Multi-Scale Hierarchical Graph Neural Network for Multimodal Survival and Recurrence Prediction in Non-Small Cell Lung Cancer*. Journal of Medical Imaging.

## 🏗️ Architecture
