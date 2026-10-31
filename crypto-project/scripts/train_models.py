"""
Train the three models the dashboard loads (SMOTE + graph features) and save them
to models/*_smote_v1_graph.pkl. Same recipe as notebooks/2.0-model-training_v1.ipynb
(the recipe itself lives in app/training.py, shared with the dashboard),
so a fresh clone can rebuild the models without Git LFS or a release download.

Prerequisites:
    python scripts/download_data.py        # raw Elliptic files
    run notebooks/1.0-data-prep.ipynb      # creates data/processed/..._v1_graph.csv

Usage (from repo root):
    python scripts/train_models.py [--out models]
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
from training import model_filename, train_models  # noqa: E402

DATA = ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'models'))
    out = Path(ap.parse_args().out)

    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}. Run scripts/download_data.py and notebooks/1.0-data-prep.ipynb first.")

    df = pd.read_csv(DATA).rename(columns={'feat_0': 'time_step'})
    out.mkdir(parents=True, exist_ok=True)
    for name, model in train_models(df).items():
        path = out / model_filename(name)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  saved {path}")


if __name__ == '__main__':
    main()
