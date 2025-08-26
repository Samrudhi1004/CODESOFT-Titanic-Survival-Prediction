# src/train.py
import argparse
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.preprocess import load_data, add_features, build_preprocessor


def main(data_path: str, model_out: str, model_type: str = 'rf'):
    df = load_data(data_path)
    df = add_features(df)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    print('Training...')
    pipeline.fit(X_train, y_train)

    print('Evaluating on test set...')
    y_pred = pipeline.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('\nClassification report:\n', classification_report(y_test, y_pred))

    # Save pipeline
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f'Model pipeline saved to: {model_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/Titanic-Dataset.csv')
    parser.add_argument('--model-out', type=str, default='models/rf_pipeline.joblib')
    parser.add_argument('--model', type=str, choices=['rf', 'logreg'], default='rf')
    args = parser.parse_args()
    main(args.data_path, args.model_out, args.model)
