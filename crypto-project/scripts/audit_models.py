"""
Audit 2: are the models genuinely predictive, and are the reported numbers trustworthy?

 1. independent metric recomputation of the shipped models (no app code)
 2. baselines: always-licit, random scores, prevalence-level PR-AUC
 3. sanity: train with shuffled labels -> must collapse to ~chance
 4. bootstrap 95% CIs on the test set
 5. ablation: all features vs no-graph vs local-only (feat_1..feat_93)
 6. drift: performance before / after time step 43 (dark-market shutdown)
"""
import pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, matthews_corrcoef)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv').rename(columns={'feat_0': 'time_step'})
df = df.drop(columns=['class'])
cols = [c for c in df.columns if c not in ('label', 'txId', 'time_step')]
tr, te = df[df.time_step <= 34], df[df.time_step > 34]
Xte, yte = te[cols], te.label.values
rng = np.random.default_rng(0)

def row(y, p, thr=0.5):
    yh = p >= thr
    return dict(Precision=precision_score(y, yh, zero_division=0), Recall=recall_score(y, yh),
                F1=f1_score(y, yh), MCC=matthews_corrcoef(y, yh), ROC_AUC=roc_auc_score(y, p),
                PR_AUC=average_precision_score(y, p))

print('=== 1. Shipped models, independent recomputation (test = steps 35-49, thr 0.5) ===')
names = {'Logistic Regression': 'logistic_regression', 'Random Forest': 'random_forest', 'XGBoost': 'xgboost'}
probs = {}
for n, f in names.items():
    m = pickle.load(open(ROOT / f'models/{f}_smote_v1_graph.pkl', 'rb'))
    probs[n] = m.predict_proba(Xte)[:, 1]
print(pd.DataFrame({n: row(yte, p) for n, p in probs.items()}).T.round(3).to_string())

print('\n=== 2. Baselines ===')
print(f'test prevalence = {yte.mean():.3f}  -> PR-AUC of a random scorer ~ {yte.mean():.3f}')
rand = rng.random(len(yte))
print('random scores     PR-AUC %.3f  ROC-AUC %.3f' % (average_precision_score(yte, rand), roc_auc_score(yte, rand)))
print('always-licit      accuracy %.3f  recall 0.000 (accuracy alone is meaningless here)' % (1 - yte.mean()))

print('\n=== 3. Shuffled-label sanity (RF, same pipeline) ===')
Xtr, ytr = tr[cols], tr.label.values
ysh = rng.permutation(ytr)
Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ysh)
p = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(Xr, yr).predict_proba(Xte)[:, 1]
print('shuffled labels -> test PR-AUC %.3f (prevalence %.3f), ROC-AUC %.3f  (should be ~chance)' %
      (average_precision_score(yte, p), yte.mean(), roc_auc_score(yte, p)))

print('\n=== 4. Bootstrap 95% CI (1000 resamples of the test set) ===')
for n in ['Random Forest', 'XGBoost']:
    p = probs[n]; f1s, aps = [], []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        f1s.append(f1_score(yte[i], p[i] >= .5)); aps.append(average_precision_score(yte[i], p[i]))
    print(f'{n:14s} F1 {np.mean(f1s):.3f} [{np.percentile(f1s,2.5):.3f}, {np.percentile(f1s,97.5):.3f}]'
          f'   PR-AUC {np.mean(aps):.3f} [{np.percentile(aps,2.5):.3f}, {np.percentile(aps,97.5):.3f}]')

print('\n=== 5. Ablation (XGBoost / RF, SMOTE, train<=34) ===')
local = [f'feat_{i}' for i in range(1, 94)]
sets = {'all (167)': cols, 'no graph degree (165)': [c for c in cols if 'degree' not in c],
        'local only (93)': local, 'degree only (2)': ['in_degree', 'out_degree']}
res = {}
for sname, cs in sets.items():
    Xr, yr = SMOTE(random_state=42).fit_resample(tr[cs], tr.label)
    for mname, mk in [('XGB', lambda: XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)),
                      ('RF', lambda: RandomForestClassifier(100, random_state=42, n_jobs=-1))]:
        p = mk().fit(Xr, yr).predict_proba(te[cs])[:, 1]
        res[(sname, mname)] = {k: v for k, v in row(yte, p).items() if k in ('Precision', 'Recall', 'F1', 'PR_AUC')}
print(pd.DataFrame(res).T.round(3).to_string())

print('\n=== 6. Drift: XGBoost by period ===')
p = probs['XGBoost']; ts = te.time_step.values
for lab, mask in [('steps 35-42', ts <= 42), ('steps 43-49', ts >= 43)]:
    r = row(yte[mask], p[mask]); print(f'{lab}: n={mask.sum():5d} illicit={yte[mask].sum():4d}  P={r["Precision"]:.3f} R={r["Recall"]:.3f} F1={r["F1"]:.3f} PR-AUC={r["PR_AUC"]:.3f}')
