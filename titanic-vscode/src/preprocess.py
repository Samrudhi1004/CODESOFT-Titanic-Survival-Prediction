# src/preprocess.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple engineered features and drop clearly irrelevant columns."""
    df = df.copy()

    # Family size and alone flag
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Drop columns that we won't use directly
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c, axis=1)

    return df


def build_preprocessor():
    """Return a ColumnTransformer that handles numeric and categorical features."""
    numerical_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    categorical_cols = ['Sex', 'Embarked']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor
