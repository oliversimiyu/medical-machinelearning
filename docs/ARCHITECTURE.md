# Project Architecture

## Overview

This project implements a medical insurance fraud detection system using Graph Neural Networks (GNN) and Graph SMOTE for handling imbalanced data.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  • Raw Claims Data                                           │
│  • Synthetic Data Generation                                 │
│  • Data Preprocessing & Feature Engineering                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Graph Construction                         │
├─────────────────────────────────────────────────────────────┤
│  • Node Creation (Claims as nodes)                           │
│  • Edge Creation (Provider relationships + KNN similarity)   │
│  • Feature Attribution                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Graph SMOTE (Balancing)                     │
├─────────────────────────────────────────────────────────────┤
│  • Minority Class Identification                             │
│  • Synthetic Node Generation                                 │
│  • Graph Structure Preservation                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  GNN Models     │  │ Traditional ML  │                  │
│  │  • GCN          │  │ • Random Forest │                  │
│  │  • GraphSAGE    │  │ • XGBoost       │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                ↓              ↓                               │
│         ┌──────────────────────────┐                         │
│         │  Stacking Ensemble       │                         │
│         │  (Meta-learner)          │                         │
│         └──────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation & Results                      │
├─────────────────────────────────────────────────────────────┤
│  • Performance Metrics                                       │
│  • Visualizations (ROC, Confusion Matrix)                    │
│  • Model Comparison                                          │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Data Layer
- **Data Generator**: Creates synthetic medical claims data
- **Feature Engineering**: Builds traditional and relational features
- **Preprocessor**: Standardizes and prepares data for modeling

### Graph Layer
- **Graph Builder**: Constructs heterogeneous graph from claims
- **Graph SMOTE**: Balances minority class while preserving topology

### Model Layer
- **GNN Models**: GCN and GraphSAGE for learning graph representations
- **Traditional Models**: Random Forest, XGBoost for baseline comparison
- **Stacking Ensemble**: Combines multiple models for improved performance

### Evaluation Layer
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: ROC curves, confusion matrices, feature importance
- **Reports**: Comprehensive performance analysis

## Technology Stack

- **Graph Processing**: NetworkX, PyTorch Geometric
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch
- **Data Analysis**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
