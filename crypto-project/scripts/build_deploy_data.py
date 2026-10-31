"""
Write the compact, lossless data bundle the hosted dashboard reads when the
full CSVs under data/ are not available (e.g. on Streamlit Community Cloud).

    deploy_data/elliptic_v1_graph.parquet   labelled transactions + graph features (~22 MB)
    deploy_data/elliptic_txs_edgelist.parquet   transaction edges (~2 MB)

Usage (from repo root, after notebooks/1.0-data-prep.ipynb):
    python scripts/build_deploy_data.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'deploy_data'


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv').drop(columns=['class'], errors='ignore')
    df.to_parquet(OUT / 'elliptic_v1_graph.parquet', compression='zstd', compression_level=19, index=False)
    assert pd.read_parquet(OUT / 'elliptic_v1_graph.parquet').equals(df), 'parquet round-trip is not lossless'
    edges = pd.read_csv(ROOT / 'data/raw/elliptic_txs_edgelist.csv')
    edges.to_parquet(OUT / 'elliptic_txs_edgelist.parquet', compression='zstd', compression_level=19, index=False)
    for f in sorted(OUT.glob('*.parquet')):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
