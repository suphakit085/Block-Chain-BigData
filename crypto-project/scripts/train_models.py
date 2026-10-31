"""
Train the three models the dashboard loads (SMOTE + graph features) and save them
to models/*_smote_v1_graph.pkl. Same recipe as notebooks/2.0-model-training_v1.ipynb,
so a fresh clone can rebuild the models without Git LFS or a release download.

Prerequisites:
    python scripts/download_data.py        # raw Elliptic files
    run notebooks/1.0-data-prep.ipynb      # creates data/processed/..._v1_graph.csv

Usage (from repo root):
    python scripts/train_models.py [--out models]
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv'
TRAIN_END = 34  # time steps 1-34 train, 35-49 are held out for the dashboard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'models'))
    out = Path(ap.parse_args().out)

    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}. Run scripts/download_data.py and notebooks/1.0-data-prep.ipynb first.")

    df = pd.read_csv(DATA).rename(columns={'feat_0': 'time_step'}).drop(columns=['class'], errors='ignore')
    train = df[df['time_step'] <= TRAIN_END]
    X = train.drop(columns=['label', 'txId', 'time_step'])
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, train['label'])
    print(f"Train rows: {len(X):,} -> {len(X_res):,} after SMOTE")

    models = {
        # Unscaled LR is unusable on this data (test PR-AUC ~0.19); scaling + stronger L2 -> ~0.36
        'Logistic Regression': make_pipeline(
            StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_res, y_res)
        path = out / f"{name.replace(' ', '_').lower()}_smote_v1_graph.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  saved {path}")


if __name__ == '__main__':
    main()
