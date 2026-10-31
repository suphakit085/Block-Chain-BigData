"""
Choose the alert threshold on a validation window instead of the test set,
and compare imbalance strategies (SMOTE vs class weights) incl. calibration.

Split (time-ordered, no shuffling):
    train       time_step  1-24
    validation  time_step 25-34   -> pick threshold / compare strategies
    test        time_step 35-49   -> reported once, never used for tuning

Usage (from repo root):
    python scripts/validate_threshold.py

"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv').rename(columns={'feat_0': 'time_step'})
df = df.drop(columns=['class'], errors='ignore')
cols = [c for c in df.columns if c not in ('label', 'txId', 'time_step')]

tr, va, te = (df[df.time_step <= 24], df[(df.time_step > 24) & (df.time_step <= 34)], df[df.time_step > 34])
Xtr, ytr, Xva, yva, Xte, yte = tr[cols], tr.label, va[cols], va.label, te[cols], te.label
pos_w = (ytr == 0).sum() / (ytr == 1).sum()


def make(name):
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=2000, random_state=42)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    return XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)


def fit(name, strategy):
    m = make(name)
    if strategy == 'SMOTE':
        Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
        return m.fit(Xr, yr)
    if strategy == 'class_weight':
        if name == 'XGBoost':
            m.set_params(scale_pos_weight=pos_w)
        else:
            m.set_params(class_weight='balanced')
        return m.fit(Xtr, ytr)
    # 'plain' (no resampling) -- probabilities are calibrated on validation afterwards
    return m.fit(Xtr, ytr)


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


rows = []
for name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
    for strategy in ['SMOTE', 'class_weight', 'plain+isotonic']:
        base = fit(name, 'plain' if strategy == 'plain+isotonic' else strategy)
        pva, pte = base.predict_proba(Xva)[:, 1], base.predict_proba(Xte)[:, 1]
        if strategy == 'plain+isotonic':
            iso = IsotonicRegression(out_of_bounds='clip').fit(pva, yva)
            pva, pte = iso.predict(pva), iso.predict(pte)
        thr = best_threshold(yva, pva)
        rows.append({
            'Model': name, 'Strategy': strategy, 'Val threshold': round(thr, 3),
            'Val F1': f1_score(yva, pva >= thr),
            'Test F1 @val-thr': f1_score(yte, pte >= thr),
            'Test F1 @0.5': f1_score(yte, pte >= 0.5),
            'Test PR-AUC': average_precision_score(yte, pte),
            'Test Brier': brier_score_loss(yte, pte),
        })
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT / 'reports/threshold_validation.csv', index=False)
(ROOT / 'models').mkdir(exist_ok=True)
print(out.round(3).to_string(index=False))
