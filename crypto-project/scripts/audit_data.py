"""Audit 1: is the processed dataset faithful to the raw Elliptic files?"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
raw = ROOT / 'data/raw'
proc = pd.read_csv(ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv')
cls = pd.read_csv(raw / 'elliptic_txs_classes.csv')
edges = pd.read_csv(raw / 'elliptic_txs_edgelist.csv')
feat = pd.read_csv(raw / 'elliptic_txs_features.csv', header=None)
feat.columns = ['txId'] + [f'feat_{i}' for i in range(166)]
ok = lambda c, msg: print(('PASS ' if c else 'FAIL ') + msg)

print('class values in raw:', cls['class'].value_counts().to_dict())
known = cls[cls['class'] != 'unknown']
print('raw rows', len(feat), '| labelled', len(known), '| illicit(class 1)', (known['class'].astype(str) == '1').sum())
ok(len(proc) == len(known), f'processed rows ({len(proc)}) == labelled raw rows ({len(known)})')
ok(proc.txId.is_unique, 'txId unique')
ok(set(proc.txId) == set(known.txId), 'same txId set as labelled raw')
ok(proc.drop(columns=['class']).isna().sum().sum() == 0, 'no NaN')

# Label mapping: in Elliptic, class "1" = illicit, "2" = licit
m = proc.merge(cls.rename(columns={'class': 'raw_class'}), on='txId')
ok(((m['raw_class'].astype(str) == '1') == (m.label == 1)).all(), 'label==1 exactly where raw class == 1 (illicit)')
print('illicit share:', round(proc.label.mean(), 4), '| illicit', int(proc.label.sum()), 'licit', int((proc.label == 0).sum()))

# Feature values identical to raw
fc = [f'feat_{i}' for i in range(166)]
j = proc.merge(feat, on='txId', suffixes=('', '_r'))
ok(all(np.allclose(j[c], j[c + '_r']) for c in fc[:5] + fc[100:105] + fc[-3:]), 'sampled feature columns identical to raw')
ok(proc.feat_0.between(1, 49).all() and proc.feat_0.nunique() == 49, 'feat_0 is time step 1..49')

# Degree features
indeg = edges.txId2.value_counts(); outdeg = edges.txId1.value_counts()
ok((proc.in_degree.values == proc.txId.map(indeg).fillna(0).values).all(), 'in_degree matches edge list')
ok((proc.out_degree.values == proc.txId.map(outdeg).fillna(0).values).all(), 'out_degree matches edge list')
ts = dict(zip(feat.txId, feat.feat_0))
ok((edges.txId1.map(ts) == edges.txId2.map(ts)).all(), 'every edge stays inside one time step (degree cannot leak future)')

print('\nillicit per time step (train<=34 | test>34):')
g = proc.groupby('feat_0').label.agg(['sum', 'count'])
print('train illicit', int(g.loc[:34, 'sum'].sum()), '/', int(g.loc[:34, 'count'].sum()),
      '| test illicit', int(g.loc[35:, 'sum'].sum()), '/', int(g.loc[35:, 'count'].sum()))
print('steps with 0 illicit:', g.index[g['sum'] == 0].tolist())
