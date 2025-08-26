# Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using
machine learning (Logistic Regression, Random Forest).

## Steps:

1. `notebooks/titanic_exploration.ipynb` → Exploratory Data Analysis (EDA).
2. `src/preprocess.py` → Preprocessing functions & feature engineering.
3. `src/train.py` → Train a model and save pipeline.
4. `src/evaluate.py` → Load model and evaluate on dataset.

## Results:

- Random Forest accuracy: ~81-83%
- Logistic Regression accuracy: ~78-80%

## How to run:

```bash
pip install -r requirements.txt
python -m src.train --data-path data/Titanic-Dataset.csv --model-out models/rf_pipeline.joblib
python -m src.evaluate --model-path models/rf_pipeline.joblib --data-path data/Titanic-Dataset.csv
```
