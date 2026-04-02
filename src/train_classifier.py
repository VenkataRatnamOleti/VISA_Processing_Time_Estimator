"""
Train a classifier to predict whether a visa case is Certified (approved) or Denied (rejected).

This script reads `src/data/VisaFile.csv`, trains a RandomForestClassifier using a
ColumnTransformer pipeline (scaling numeric features and one-hot encoding categoricals),
evaluates performance on a held-out test set, and saves the fitted pipeline to
`models/visa_status_model.pkl`.

Usage: run from repository root (or run this file directly):
	python src/train_classifier.py
"""

from pathlib import Path
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "src" / "data" / "VisaFile.csv"
MODEL_PATH = ROOT / "models" / "visa_status_model.pkl"


def load_data(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	return df


def preprocess_and_train(df: pd.DataFrame):
	# target: case_status -> Certified=1, Denied=0
	df = df.copy()
	df['case_status'] = df['case_status'].map({'Certified': 1, 'Denied': 0})

	# Drop identifier
	if 'case_id' in df.columns:
		df = df.drop(columns=['case_id'])

	y = df['case_status']
	X = df.drop(columns=['case_status'])

	# Normalize some obvious binary text columns to 0/1
	binary_cols = [c for c in X.columns if X[c].dtype == object and set(X[c].dropna().unique()) <= {"Y", "N"}]
	for c in binary_cols:
		X[c] = X[c].map({'Y': 1, 'N': 0})

	# Choose numeric and categorical features
	numeric_features = [c for c in X.columns if X[c].dtype in [int, float] or pd.api.types.is_numeric_dtype(X[c])]
	# After mapping binary cols above some binary_cols will be numeric; remove them from categorical set.
	categorical_features = [c for c in X.columns if c not in numeric_features]

	# Build preprocessing
	numeric_transformer = StandardScaler()
	categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_transformer, numeric_features),
			('cat', categorical_transformer, categorical_features),
		],
		remainder='drop'
	)

	clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

	pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")
	pipe.fit(X_train, y_train)

	y_pred = pipe.predict(X_test)
	print("Test accuracy:", accuracy_score(y_test, y_pred))
	print("Classification report:\n", classification_report(y_test, y_pred))

	# Ensure models folder exists
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipe, MODEL_PATH)
	print(f"Saved trained classifier pipeline to: {MODEL_PATH}")


def main():
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

	df = load_data(DATA_PATH)
	preprocess_and_train(df)


if __name__ == '__main__':
	main()

