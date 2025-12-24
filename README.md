# Medical Insurance Fraud Detection using Graph Neural Networks

A comprehensive research project comparing baseline models, SMOTE-enhanced models, and Graph Neural Network (GNN) approaches for detecting fraudulent medical insurance claims.

## 📋 Project Overview

This project implements and compares three distinct approaches to medical insurance fraud detection:

1. **Baseline Model** - Traditional tabular features with Stacking Ensemble
2. **SMOTE Model** - Relational patterns with SMOTE for class imbalance
3. **Proposed GNN Model** - Graph-based representation with Graph SMOTE and GNN architectures

## 🏗️ Project Structure

> **Note:** This is a **pure notebook-based research project**. All implementations (models, features, evaluation) are self-contained within the Jupyter notebooks.

```
Medical-machine learning/
│
├── notebooks/                          # 🎯 ALL IMPLEMENTATIONS HERE
│   ├── fraud_detection_base_model.ipynb      # Baseline model (complete)
│   ├── fraud_detection_smote_model.ipynb     # SMOTE model (complete)
│   └── fraud_detection_gnn_model.ipynb       # GNN model (complete)
│
├── src/                               # 📦 Optional data utilities only
│   └── data/                         # Data generation helpers
│       ├── data_generator.py        # Synthetic data generation
│       └── graph_builder.py         # Graph construction utilities
│
├── data/                              # Dataset storage
├── models/                            # Saved trained models (.pkl, .pth)
├── results/                           # Experiment outputs
│   ├── figures/                      # Visualizations and plots
│   └── reports/                      # Analysis reports
│
├── tests/                             # Unit tests
│   ├── test_data_generator.py
│   └── test_graph_smote.py
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md              # System architecture
│   └── USAGE.md                     # Usage guide
│
├── configs/                           # Configuration files
│   └── config.yaml                  # Model hyperparameters
│
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## 🚀 Key Features

### Baseline Model
- Traditional tabular feature representation
- Basic feature engineering (3 simple ratio features)
- Stacking Ensemble (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Logistic Regression meta-learner
- Class-weight balancing

### SMOTE Model
- Advanced relational pattern features (35+ features)
- Provider behavior patterns
- Temporal claim patterns
- Cost anomaly indicators
- Standard SMOTE for class imbalance
- Stacking Ensemble with enhanced features

### Proposed GNN Model
- Graph-based representation of claims network
- Custom Graph SMOTE implementation
- Graph Convolutional Network (GCN)
- GraphSAGE architecture
- GNN embeddings combined with traditional features
- Hybrid stacking ensemble

## 📊 Models Implemented

| Model | Approach | Key Technique | Class Imbalance Handling |
|-------|----------|---------------|-------------------------|
| Baseline | Traditional ML | Stacking Ensemble | Class weights |
| SMOTE | Enhanced Features | Relational Patterns | Standard SMOTE |
| GNN (Proposed) | Graph-based | Graph Neural Networks | Graph SMOTE |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for GNN models)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Medical-machine\ learning
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install package in development mode:**
```bash
pip install -e .
```

5. **For GNN support (optional):**
```bash
# CPU version
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
```

## 📈 Usage

### Primary Workflow: Run the Notebooks

All implementations are **self-contained in the notebooks**. Simply open and run them:

1. **Baseline Model:**
```bash
jupyter notebook notebooks/fraud_detection_base_model.ipynb
```
- Traditional tabular features
- Stacking ensemble with class weights
- Complete implementation from data to evaluation

2. **SMOTE Model:**
```bash
jupyter notebook notebooks/fraud_detection_smote_model.ipynb
```
- 35+ relational pattern features
- SMOTE for class imbalance
- Complete implementation with advanced features

3. **GNN Model (Proposed):**
```bash
jupyter notebook notebooks/fraud_detection_gnn_model.ipynb
```
- Graph construction and Graph SMOTE
- GCN and GraphSAGE implementations
- Complete GNN-based fraud detection

### Optional: Data Generation Utilities

The `src/data/` folder contains optional helpers for data generation:

```python
# Generate synthetic data (optional - also implemented in notebooks)
from src.data.data_generator import generate_synthetic_data
X, y = generate_synthetic_data(n_samples=10000, fraud_ratio=0.05)

# Build graph structure (optional - also implemented in notebooks)
from src.data.graph_builder import construct_claim_graph
G = construct_claim_graph(X, y, k_neighbors=5)
```

**Note:** These utilities are optional. All notebooks have complete implementations without external dependencies.

## 📊 Dataset

The project uses synthetic medical insurance claims data with the following features:

- `claim_amount`: Dollar amount of the claim
- `patient_age`: Age of the patient
- `num_procedures`: Number of medical procedures
- `hospital_stay_days`: Length of hospital stay
- `num_previous_claims`: Historical claim frequency
- `provider_claim_count`: Provider activity level
- `diagnosis_complexity`: Complexity score
- `treatment_cost_ratio`: Cost-to-treatment ratio
- `claim_processing_time`: Processing duration
- `geographic_risk_score`: Regional risk indicator

**Target Variable:** `is_fraud` (0 = Legitimate, 1 = Fraudulent)

## 🎯 Performance Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection precision
- **Recall**: Fraud detection coverage
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Average Precision**: Area under PR curve

## 🔬 Research Contributions

1. **Graph-based fraud detection framework** for insurance claims
2. **Custom Graph SMOTE algorithm** for imbalanced graph data
3. **Hybrid GNN-traditional ML approach** using stacking ensembles
4. **Comprehensive comparison** of three distinct methodologies:
   - Baseline: Traditional features (3 features) + Stacking
   - SMOTE: Relational features (35+ features) + SMOTE + Stacking
   - GNN: Graph representation + Graph SMOTE + GNN + Stacking
5. **Fully documented notebooks** with complete implementation

## 📝 Methodology

### 1. Data Preparation
- Load and preprocess medical insurance claims
- Handle missing values and outliers
- Split into training and test sets (80/20)

### 2. Feature Engineering
- **Baseline**: Basic ratio features (3 features)
- **SMOTE**: Relational patterns (35+ features)
- **GNN**: Graph construction from claim relationships

### 3. Class Imbalance Handling
- **Baseline**: Class weights in models
- **SMOTE**: Synthetic minority oversampling
- **GNN**: Graph SMOTE preserving topology

### 4. Model Training
- Train individual base learners
- Build stacking ensembles
- Optimize hyperparameters

### 5. Evaluation
- Compare models on test set
- Analyze performance metrics
- Generate visualizations

## 📊 Results

Results are generated in the notebooks and can be saved to:
- Visualizations: `results/figures/`
- Detailed reports: `results/reports/`

Each notebook includes comprehensive evaluation with:
- Classification reports
- Confusion matrices
- ROC curves
- Performance comparisons

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Your Name** - Medical Insurance Fraud Detection Research

## 🙏 Acknowledgments

- PyTorch Geometric for GNN implementations
- imbalanced-learn for SMOTE baseline
- scikit-learn for ensemble methods
- NetworkX for graph processing
- Graph Neural Network architectures inspired by PyTorch Geometric
- SMOTE implementation based on imbalanced-learn
- Ensemble methods from scikit-learn

## 📚 References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. NeurIPS.
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note:** This is a research project for educational purposes. The data used is synthetic and generated for demonstration.
