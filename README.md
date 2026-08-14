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
┌─────────────────────────────────────────────────────────────┐
│ MS-HGNN Framework │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────────────────────┼─────────────────────────────┐
│ │ │
▼ ▼ ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ Level 1 │ │ Level 2 │ │ Level 3 │
│ Cross-Modal │ │ Heterogeneous │ │ Uncertainty-Aware │
│ Attention │────────▶│ Graphs + │────────▶│ Fusion + │
│ Fusion │ │ Semantic Attention│ │ Monte Carlo │
│ │ │ │ │ Dropout │
└───────────────────┘ └───────────────────┘ └───────────────────┘
│
▼
┌───────────────────────┐
│ Multi-Task Output │
│ • Survival Prediction│
│ • Recurrence Classif.│
│ • Uncertainty Scores │
└───────────────────────┘

text

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/ImamDad/MS-HGNN.git
cd MS-HGNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
🚀 Quick Start
1. Data Preparation
bash
# Download data from TCIA
python scripts/download_data.py --data_dir ./data --dataset tcia

# Preprocess data
python scripts/preprocess_data.py --data_dir ./data --output_dir ./data/processed
2. Training
bash
# Train from scratch
python main.py train --data_dir ./data/processed --epochs 100 --batch_size 16 --device cuda

# Train with custom config
python main.py train --config configs/custom_config.json --data_dir ./data/processed
3. Evaluation
bash
# Evaluate on test set
python main.py eval --model_path ./checkpoints/best_model.pt --data_dir ./data/processed --output_dir ./results

# Run external validation on TCGA-LUAD
python main.py eval --model_path ./checkpoints/best_model.pt --data_dir ./data/processed --external_dir ./data/tcga_luad
4. Jupyter Notebooks
Explore the notebooks for interactive demos:

bash
jupyter notebook notebooks/
📊 Results
Performance Summary
Metric	MS-HGNN	Best Baseline (FGCN)	p-value
C-index (Survival)	0.85 (0.82-0.88)	0.82 (0.79-0.85)	<0.001
AUC (Recurrence)	0.89 (0.86-0.92)	0.84 (0.81-0.87)	<0.001
Hazard Ratio	4.15 (3.12-5.52)	~2.0 (TNM staging)	<0.001
Model Compression Results
Version	Parameters	Memory	Inference	C-index
Full MS-HGNN	28.5M	9.8GB	85ms	0.85
Pruned + Quantized	14.2M	1.3GB	28ms	0.83
🧪 Experiments and Reproducibility
To reproduce the paper's results:

bash
# Run full training pipeline
python scripts/run_experiment.py --config configs/paper_config.json

# Run benchmark against baselines
python scripts/run_benchmark.py --data_dir ./data/processed --output_dir ./benchmarks

# Run ablation studies
python scripts/run_ablation.py --data_dir ./data/processed --output_dir ./ablation_results
📁 Project Structure
text
MS-HGNN/
├── config.py              # Configuration management
├── data_loader.py         # Data loading and preprocessing
├── train.py               # Training loop
├── evaluate.py            # Evaluation and validation
├── main.py                # Main entry point
├── models/
│   ├── encoders.py        # Modality-specific encoders
│   ├── cross_modal_attention.py  # Level 1 fusion
│   ├── heterogeneous_graph.py    # Level 2 graph construction
│   ├── semantic_attention.py     # Semantic attention mechanism
│   ├── uncertainty_fusion.py     # Level 3 uncertainty fusion
│   └── ms_hgnn.py         # Complete MS-HGNN architecture
├── utils/
│   ├── preprocessing.py   # Data preprocessing utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── visualization.py   # Visualization utilities
│   └── augmentation.py    # Data augmentation
├── notebooks/             # Jupyter notebooks for demos
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── docs/                  # Documentation
🔬 Model Interpretation
Semantic Attention Weights
The model learns patient-specific importance weights for three biologically-informed meta-paths:

Meta-Path	Mean Weight	Biological Validation
Immune	0.42	CD8+ infiltration: ρ = 0.67
Proliferation	0.35	Ki-67 index: ρ = 0.59
Treatment Response	0.23	DNA repair signature: ρ = 0.52
Feature Importance
Top predictive features identified by MS-HGNN:

Feature	Modality	Significance
GLCM Correlation	CT	Tumor heterogeneity
GLCM Contrast	CT	Texture irregularity
SUVmax	PET	Glucose metabolism
Metabolic Tumor Volume	PET	Tumor burden
Immune Response Pathway	Genomic	Immune infiltration
🌐 External Validation
MS-HGNN was validated on TCGA-LUAD (n=69):

Setting	C-index	AUC
Direct Transfer	0.78	0.81
Fine-tuned	0.81	0.84
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 Citation
If you use this code in your research, please cite:

bibtex
@article{dad2026mshgnn,
  title={MS-HGNN: Interpretable Multi-Scale Hierarchical Graph Neural Network for Multimodal Survival and Recurrence Prediction in Non-Small Cell Lung Cancer},
  author={Dad, Imam and He, Jianfeng},
  journal={Journal of Medical Imaging},
  year={2026}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
National Natural Science Foundation of China (Grant No. 82160347)

TCIA and TCGA for providing the public datasets

Kunming University of Science and Technology for computational resources

Chinese Government Scholarship Council (CSC)

📧 Contact
Imam Dad - GitHub

Jianfeng He (Corresponding Author) - jfenghe@kust.edu.cn

⭐ Star History
If you find this project useful, please consider giving it a star ⭐!

https://api.star-history.com/svg?repos=ImamDad/MS-HGNN&type=Date
