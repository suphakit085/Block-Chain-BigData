"""
Can retraining recover performance after the step-43 regime change?

Walk-forward evaluation on test steps 35-49. For each step t the model is trained
only on labelled data from steps < t - delay (delay = label latency, default 1 step),
then predicts step t. Strategies:
    static      train on steps 1-34 once (what the dashboard ships)
    expanding   all steps up to t-1-delay
    sliding-W   only the last W steps before t-delay
Model: XGBoost with scale_pos_weight (fast, comparable across strategies).
Writes reports/drift_retraining.csv.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

DELAY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv').rename(columns={'feat_0': 'time_step'})
cols = [c for c in df.columns if c not in ('label', 'txId', 'time_step', 'class')]

def fit(train):
    spw = (train.label == 0).sum() / max((train.label == 1).sum(), 1)
    return XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, scale_pos_weight=spw
                         ).fit(train[cols], train.label)

static = fit(df[df.time_step <= 34])
strategies = {'static (1-34)': None, 'expanding': None, 'sliding-5': 5, 'sliding-10': 10}
preds = {k: [] for k in strategies}
for t in range(35, 50):
    te = df[df.time_step == t]
    last = t - 1 - DELAY                       # newest step whose labels are known
    for name, w in strategies.items():
        if name.startswith('static'):
            m = static
        else:
            lo = 1 if name == 'expanding' else max(1, last - w + 1)
            m = fit(df[(df.time_step >= lo) & (df.time_step <= last)])
        preds[name].append(pd.DataFrame({'t': t, 'y': te.label.values, 'p': m.predict_proba(te[cols])[:, 1]}))

rows = []
for name, parts in preds.items():
    d = pd.concat(parts)
    for per, mask in [('35-42', d.t <= 42), ('43-49', d.t >= 43), ('all 35-49', d.t >= 35)]:
        x = d[mask]
        rows.append(dict(Strategy=name, Period=per, Precision=precision_score(x.y, x.p >= .5, zero_division=0),
                         Recall=recall_score(x.y, x.p >= .5), F1=f1_score(x.y, x.p >= .5),
                         PR_AUC=average_precision_score(x.y, x.p)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / 'reports/drift_retraining.csv', index=False)
print(f'label delay = {DELAY} step(s)')
print(out.round(3).to_string(index=False))
print('\nper-step recall (steps 41-49):')
print(pd.DataFrame({n: pd.concat(p).groupby('t').apply(lambda x: recall_score(x.y, x.p >= .5)).loc[41:] for n, p in preds.items()}).round(2).to_string())

print('\nthreshold sweep on steps 43-49 (does better ranking let a lower threshold recover recall?):')
sw = []
for name, parts in preds.items():
    d = pd.concat(parts); d = d[d.t >= 43]
    for thr in (0.5, 0.2, 0.1, 0.05):
        sw.append(dict(Strategy=name, thr=thr, Precision=precision_score(d.y, d.p >= thr, zero_division=0),
                       Recall=recall_score(d.y, d.p >= thr), F1=f1_score(d.y, d.p >= thr),
                       alerts=int((d.p >= thr).sum())))
print(pd.DataFrame(sw).round(3).to_string(index=False))
