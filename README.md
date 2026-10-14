# Predicting Student Dropout & Academic Success

Using the UCI **"Predict Students' Dropout and Academic Success"** dataset
(4,424 students, 8 undergraduate programs, Portugal) to explore whether
machine learning can classify a student's outcome: **Dropout**,
**Enrolled**, or **Graduate** from demographic, socioeconomic, academic
path, and macroeconomic variables available at or near enrollment.

## Notebooks

Run in order. Notebooks 3, 4 and 5 each save a results table to
`data/processed/` that notebook 7 loads, so run them before 7.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda_.ipynb` | Exploratory data analysis, class balance, collinearity, structural zeros |
| 02 | `02_logistic_baseline.ipynb` | Logistic Regression baseline (class-weighted), confusion matrix, coefficients |
| 03 | `03_feature_availability_logreg.ipynb` | Logistic Regression on Day 1 / After Semester 1 / Full Data feature sets |
| 04 | `04_random_forest.ipynb` | Same feature-availability comparison with Random Forest |
| 05 | `05_xgboost.ipynb` | Same feature-availability comparison with XGBoost, plus 5-fold CV |
| 06 | `06_gpa_trajectory.ipynb` | Secondary question: predicting semester GPA and GPA decline vs. dropout |
| 07 | `07_model_comparison.ipynb` | Logistic Regression vs. Random Forest vs. XGBoost (uses outputs of 03-05) |
| 08 | `08_sem1_mechanism_ablation.ipynb` | Which Semester-1 information (grade, approved units, activity, credits) drives the Day 1 → Semester 1 gain |
| 09 | `09_class_specific_error_analysis.ipynb` | Why Enrolled is hard to predict: where Enrolled students get misclassified and why |
| — | `assumptions_checks.ipynb` | Statistical tests and model-assumption checks for notebooks 01-07 |

## Data source

UCI Machine Learning Repository — *Predict Students' Dropout and Academic
Success*: https://archive.ics.uci.edu/dataset/697/
