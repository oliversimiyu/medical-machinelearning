# Usage Guide

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Medical-machine\ learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Data

```python
from src.data.data_generator import generate_synthetic_data

# Generate 10,000 samples with 5% fraud
X, y = generate_synthetic_data(n_samples=10000, fraud_ratio=0.05)
```

### 2. Build Graph

```python
from src.data.graph_builder import construct_claim_graph

# Construct graph from data
G = construct_claim_graph(X, y, k_neighbors=5)
```

### 3. Apply Graph SMOTE

```python
from src.models.graph_smote import GraphSMOTE

# Balance the graph
smote = GraphSMOTE(k_neighbors=5, sampling_strategy=0.8)
G_balanced = smote.fit_resample(G, X.columns.tolist())
```

### 4. Train Models

#### Traditional ML Models:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
```

#### GNN Models:
```python
from src.models.gnn_models import GCN
import torch

# Initialize and train GCN
model = GCN(input_dim=X.shape[1], hidden_dim=64)
# ... training code (see notebooks for details)
```

### 5. Evaluate

```python
from src.utils.evaluation import evaluate_model, plot_roc_curves

# Evaluate model
metrics = evaluate_model(model, X_test, y_test, model_name="Random Forest")

# Plot ROC curves
models_dict = {'RF': rf_model, 'XGB': xgb_model}
plot_roc_curves(models_dict, X_test, y_test)
```

## Running Notebooks

### Baseline Model
```bash
jupyter notebook notebooks/fraud_detection_base_model.ipynb
```

### SMOTE Model
```bash
jupyter notebook notebooks/fraud_detection_smote_model.ipynb
```

### GNN Model (Proposed)
```bash
jupyter notebook notebooks/fraud_detection_gnn_model.ipynb
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_generator.py
```

## Configuration

Edit `configs/config.yaml` to customize:
- Data generation parameters
- Model hyperparameters
- Graph construction settings
- Training configurations

## Project Structure

```
Medical-machine learning/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model implementations
│   └── utils/             # Utilities
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── data/                  # Data storage
├── models/                # Saved models
└── results/               # Outputs
```
