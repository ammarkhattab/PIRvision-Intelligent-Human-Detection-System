# PIRvision

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.35%25-brightgreen.svg)]()

**Accurate multiclass human presence detection using PIR sensor networks and advanced machine learning.**

PIRvision is an intelligent occupancy detection system designed for real-world buildings, powered by a dense network of Passive Infrared (PIR) sensors combined with environmental data. The system leverages state-of-the-art machine learning techniques, robust feature engineering, and careful data preprocessing to distinguish between three distinct occupancy states with over 99% accuracy.

## 🎯 Overview

PIRvision addresses the critical need for accurate, privacy-preserving occupancy detection in smart building applications. Unlike vision-based systems, our PIR sensor network approach ensures complete privacy while maintaining exceptional accuracy across diverse environmental conditions.

### Key Capabilities

- **Multiclass Detection**: Distinguishes between vacancy, stationary presence, and active motion states
- **High Accuracy**: Achieves 99.35% accuracy and F1-score using ensemble learning
- **Privacy-First**: No cameras or audio recording - purely thermal motion detection
- **Real-World Ready**: Tested on comprehensive datasets with temporal and spatial variations

## 🏗️ Architecture

```
┌─────────────────┐
│   PIR Sensors   │
│  (55 sensors)   │
└─────────────────┘
          │
          ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Environmental   │  ──►  │  Feature Engine │  ──►  │ ML Classifier   │
│ (Temperature)   │       │                 │       │   (Ensemble)    │
└─────────────────┘       │ • Statistical   │       └─────────────────┘
                          │ • Temporal      │                │
                          │ • Frequency     │                ▼
                          │ • PCA           │       ┌─────────────────┐
                          │ • Entropy       │       │ Occupancy State │
                          │ • Activation    │       │                 │
                          └─────────────────┘       │ 0: Vacant       │
                                                    │ 1: Stationary   │
                                                    │ 3: Active       │
                                                    └─────────────────┘

Data Flow: Sensors → Feature Engineering → ML Model → Classification
```

## 📊 Performance Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Vacant (0) | 99.9% | 99.4% | 99.7% | 1,253 |
| Stationary (1) | 97.1% | 99.4% | 98.2% | 171 |
| Active (3) | 96.6% | 99.1% | 97.8% | 106 |
| **Weighted Avg** | **99.4%** | **99.4%** | **99.4%** | **1,530** |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/PIRvision.git
   cd PIRvision
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare dataset**
   ```bash
   python scripts/download_dataset.py
   python scripts/prepare_dataset.py
   ```

### Usage

#### Training a Model
```python
from scripts.advanced_feature_engineering import PIRFeatureEngine
from sklearn.ensemble import GradientBoostingClassifier

# Load and preprocess data
feature_engine = PIRFeatureEngine()
X_train, X_test, y_train, y_test = feature_engine.prepare_data()

# Train model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
```

#### Making Predictions
```python
# Load saved model
import joblib
model = joblib.load('models/best_gradient_boosting_model.pkl')

# Predict occupancy state
prediction = model.predict(sensor_data)
# 0: Vacant, 1: Stationary, 3: Active
```

## 📁 Project Structure

```
PIRvision/
├── 📂 data/
│   ├── raw/                    # Original UCI datasets
│   └── processed/              # Cleaned and engineered features
├── 📂 notebooks/
│   ├── 01-initial-data-exploration.ipynb
│   ├── 02_comprehensive_eda.ipynb
│   └── 03_model_development.ipynb
├── 📂 scripts/
│   ├── download_dataset.py     # Data acquisition
│   ├── prepare_dataset.py      # Data preprocessing
│   ├── fix_data_issues.py      # Outlier handling
│   └── advanced_feature_engineering.py
├── 📂 models/                  # Trained models and scalers
├── 📂 reports/                 # Visualizations and analysis
│   ├── feature_importances.jpg
│   ├── model_comparison.jpg
│   └── confusion_matrix.jpg
└── 📂 presentation/            # Project presentation materials
```

## 🔬 Technical Details

### Dataset
- **Source**: UCI Machine Learning Repository - PIRvision_FoG
- **Size**: 7,651 samples with 59 features
- **Features**: 55 PIR sensors + temperature + temporal information
- **Class Distribution**: 
  - Vacant: 82% (6,274 samples)
  - Stationary: 11% (842 samples)  
  - Active: 7% (535 samples)

### Feature Engineering
- **Statistical Features**: Mean, variance, skewness, kurtosis per sensor
- **Temporal Features**: Time-based patterns and periodicity
- **Frequency Domain**: FFT-based spectral analysis
- **Dimensionality Reduction**: PCA components
- **Information Theory**: Entropy-based measures
- **Activation Patterns**: Sensor clustering and spatial analysis

### Model Architecture
The final ensemble combines multiple algorithms:
- Gradient Boosting Classifier (primary)
- Random Forest
- Support Vector Machine
- Neural Network (MLP)

### Data Preprocessing
- **Imbalance Handling**: SMOTE-Tomek resampling
- **Outlier Detection**: Statistical and temperature anomaly correction
- **Normalization**: StandardScaler for feature scaling
- **Validation**: Stratified train-test split with cross-validation

## 📈 Results & Visualizations

Our comprehensive evaluation includes:

- **Model Comparison**: Performance across 13+ algorithms
- **Feature Importance**: Analysis of most predictive sensors
- **Confusion Matrices**: Per-class prediction analysis
- **Temporal Patterns**: Occupancy trends over time
- **Sensor Heatmaps**: Spatial activation patterns

View detailed results in the [`/reports`](./reports) directory.

## 🔧 Customization

### Extending the System

1. **New Data Sources**: Modify data loaders for different sensor configurations
2. **Additional Features**: Extend feature engineering pipeline
3. **Model Variants**: Experiment with deep learning or ensemble methods
4. **Real-time Deployment**: Use saved models for live inference

### Configuration Options

```python
# Example configuration for different environments
config = {
    'sensor_count': 55,
    'sampling_rate': '1Hz',
    'features': ['statistical', 'temporal', 'frequency'],
    'model_type': 'ensemble',
    'resampling': 'smote-tomek'
}
```

## 🏢 Applications

- **Smart Buildings**: Automated lighting and HVAC control
- **Security Systems**: Intrusion detection and monitoring
- **Energy Management**: Occupancy-based power optimization
- **Space Utilization**: Meeting room and workspace analytics
- **Healthcare**: Patient monitoring and fall detection

## 📚 Research & Development

This project was developed as part of the CSC417 Machine Intelligence course and represents state-of-the-art research in privacy-preserving occupancy detection. 

### Key Contributions
- Novel feature engineering techniques for PIR sensor data
- Comprehensive evaluation of ML algorithms for occupancy detection
- Open-source implementation with full reproducibility
- Real-world validation on challenging indoor environments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

## 📄 Citation

If you use PIRvision in your research, please cite:

```bibtex
@misc{pirvision2025,
  title={PIRvision: Accurate Multiclass Human Presence Detection using PIR Sensor Networks},
  author={Khattab, Ammar Tarek},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-username/PIRvision}
}
```

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the PIRvision_FoG dataset
- CSC417 Machine Intelligence course instructors and peers
- Open source community for the excellent ML libraries

## 📞 Contact

**Ammar Tarek Khattab**  
📧 [Contact via GitHub Issues](https://github.com/your-username/PIRvision/issues)

---

<div align="center">

**PIRvision: Trusted, accurate, and privacy-friendly human presence detection for the smart buildings of tomorrow.**

⭐ Star this repository if you find it helpful!

</div>
