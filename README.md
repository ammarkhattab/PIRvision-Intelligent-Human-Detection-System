Accurate multiclass human presence detection using PIR sensor networks and advanced machine learning.

Overview
PIRvision is an intelligent occupancy detection system designed for real-world buildings, powered by a dense network of Passive Infrared (PIR) sensors along with environmental data (temperature). The system leverages state-of-the-art machine learning, robust feature engineering, and careful data handling to distinguish between three occupancy states:

Vacancy

Stationary Presence

Human Activity/Motion

Developed as part of the CSC417 Machine Intelligence course, PIRvision achieves over 99% accuracy and F1-score using ensemble models on the PIRvision_FoG open dataset.

Table of Contents
Features

Project Structure

Quick Start

Requirements

Data

Workflow & Pipeline

Key Results

Visualization & Figures

Customization & Extensions

References & Citations

License

Contact

Features
Multiclass Detection: Recognizes vacancy, stationary presence, and activity/motion states.

Robust Data Pipeline: Handles missing data, outliers, and complex temporal/spatial structures.

Advanced Feature Engineering: Includes PCA, FFT, entropy, statistical, and activation pattern features.

13+ ML Algorithms: Linear, tree-based, support vector, neural, and ensemble methods evaluated.

Automated Evaluation: Accuracy, F1, per-class breakdowns, cross-validation, and computational analysis.

Full Reproducibility: All code, models, and reports included; open dataset ready.

Project Structure
text
PIRvision/
│
├── data/
│   ├── raw/         # Original UCI datasets
│   ├── processed/   # Combined, cleaned, engineered data
│
├── notebooks/
│   ├── 01-initial-data-exploration.ipynb
│   ├── 02_comprehensive_eda.ipynb
│   ├── 03_model_development.ipynb
│
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── fix_data_issues.py
│   ├── advanced_feature_engineering.py
│
├── models/          # Saved trained model files, scalers, etc.
├── reports/
│   ├── feature_importances.jpg
│   ├── model_comparison.jpg
│   ├── best_model_confusion_matrix.jpg
│   ├── [other diagrams]
│   ├── final_report.pdf
│
├── presentation/
│   └── slides.pptx
│
├── requirements.txt
├── README.md
└── LICENSE
Quick Start
Clone the repository:

bash
git clone https://github.com/your-username/PIRvision.git
cd PIRvision
Set up Python environment:

bash
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows
pip install -r requirements.txt
Download and prepare data:

Run the provided script or download manually as per download_dataset.py.

Run analysis and train models:

Step through the notebooks in /notebooks/ to explore data, engineer features, and train models.

All major steps are modularized (exploration, EDA, feature engineering, training, evaluation, reporting).

Reproduce final results:

Use the code in /notebooks/03_model_development.ipynb and /scripts.

Requirements
Python 3.8+

Core: pandas, numpy, scikit-learn, imblearn, xgboost, matplotlib, seaborn

Other: plotly, joblib

See requirements.txt for the full list.

Data
Source: UCI Machine Learning Repository – PIRvision_FoG

Contents: 7,651 samples, 59 features (55 PIR + temp + time)

Target Classes:

0: Vacancy (~82%)

1: Stationary (~11%)

3: Activity (~7%)

See /data/ for structure and /notebooks/01-initial-data-exploration.ipynb for data loading instructions.

Workflow & Pipeline
Data Preparation:

Combine, fix, and clean raw CSVs.

Address outliers and temperature anomalies (fix_data_issues.py).

Exploratory Analysis:

Visualize distributions, sensor statistics, temporal/class balance.

Example:

Distribution bar/pie chart

PIR sensor heatmaps

Feature Engineering:

Advanced statistical, frequency (FFT), and PCA features

Activation/cluster features

Preprocessing & Splitting:

SMOTE-Tomek resampling for imbalance

Standard scaling

Model Training & Selection:

Compare 13 ML models + ensemble (Voting)

Stratified train/test split

Hyperparameter tuning (grid search/manual)

Save trained models/scalers

Evaluation & Analysis:

All relevant metrics (accuracy, F1, P/R by class)

Visualizations: confusion matrix, comparative accuracy, feature importances

Reporting & Presentation:

Full technical PDF report

Slide deck for presentation

Key Results
Best Model: Gradient Boosting Classifier

Test Set Performance:

Accuracy: 99.35%

Weighted F1-Score: 99.35%

Per-class (Vacancy, Stationary, Activity):

Vacancy: Precision 99.9%, Recall 99.4%

Stationary: Precision 97.1%, Recall 99.4%

Activity: Precision 96.6%, Recall 99.1%

Top Features:

PCA_1, Temperature_F, PIR_40, pir_par, fft_entropy

See: /reports/model_comparison.jpg, /reports/feature_importances.jpg, /reports/best_model_confusion_matrix.jpg

Visualization & Figures
Class distribution (bar, pie)

PIR sensor distribution, correlation heatmaps

Temporal class occupancy patterns

Model accuracy/training time comparisons

Feature importance plots

Confusion matrix of best model

All figures are in /reports/ and can be embedded as needed in notebooks or new reports.

Customization & Extensions
Change Data Sources: Adapt scripts to use alternate PIR or environmental datasets.

Add New Features: Extend feature engineering in /scripts/advanced_feature_engineering.py.

Try New Models: Drop in new scikit-learn models or try deep learning variants.

Deployment: Use saved model files in /models/ for inference in real-time applications or integrate into IoT firmware.

Generalize: Add temporal validation splits, transfer to new rooms/buildings, or combine with vision/ultrasonic sensors.

References & Citations
PIRvision_FoG: UCI Machine Learning Repository

[01-initial-data-exploration.pdf], [02_comprehensive_eda.pdf], [03_model_development.pdf]

CSCI417 course lectures on sensing and ML methods

License
MIT License (see LICENSE file)

Contact
Author: Ammar Tarek Khattab (ID: 202002123)

Course: CSC417 Machine Intelligence (Spring/Summer 2025)

Open for collaboration / feedback via project repository

For questions, bug reports, or commercial interest: please open an issue or reach out by email (see your repo profile).

PIRvision: Trusted, accurate, and privacy-friendly human presence detection for the environments of tomorrow.

