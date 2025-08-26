# src/evaluate.py
import joblib
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocess import load_data, add_features


def main(model_path: str, data_path: str):
    pipeline = joblib.load(model_path)
    df = load_data(data_path)
    df = add_features(df)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    preds = pipeline.predict(X)
    print('Accuracy:', accuracy_score(y, preds))
    print('\nClassification report:\n', classification_report(y, preds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/rf_pipeline.joblib')
    parser.add_argument('--data-path', type=str, default='data/Titanic-Dataset.csv')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
