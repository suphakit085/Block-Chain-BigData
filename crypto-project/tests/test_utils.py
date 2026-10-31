import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'app'))
import utils  # noqa: E402


class FakeModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs)

    def predict_proba(self, X):
        p = self.probs[X['a'].to_numpy()]  # 'a' doubles as the row id
        return np.column_stack([1 - p, p])


X = pd.DataFrame({'a': range(4)})
y = pd.Series([1, 0, 1, 0])
model = FakeModel([0.9, 0.8, 0.4, 0.1])


def test_predict_with_threshold_is_inclusive():
    assert utils.predict_with_threshold(model, X, 0.4).tolist() == [1, 1, 1, 0]
    assert utils.predict_with_threshold(model, X, 0.5).tolist() == [1, 1, 0, 0]


def test_confusion_matrix_layout():
    cm = utils.get_confusion_matrix_data(model, X, y, 0.5)
    assert cm.tolist() == [[1, 1], [1, 1]]  # [[TN, FP], [FN, TP]]


def test_precision_at_k_skips_k_larger_than_data():
    df = utils.get_precision_at_k(model, X, y, ks=(2, 100))
    assert df['Review budget (K)'].tolist() == [2]
    assert df['Fraud found'].iloc[0] == 1
    assert df['Share of all fraud caught'].iloc[0] == 0.5


def test_fraud_class_shap_handles_all_shap_formats():
    arr = np.arange(24).reshape(2, 4, 3)
    assert utils._fraud_class_shap(arr).shape == (2, 4)
    assert (utils._fraud_class_shap(arr) == arr[:, :, 1]).all()
    assert utils._fraud_class_shap([arr[:, :, 0], arr[:, :, 1]]).shape == (2, 4)
    assert utils._fraud_class_shap(arr[:, :, 0]).shape == (2, 4)


def test_metrics_by_time_step_groups_correctly():
    df = pd.DataFrame({'a': range(4), 'time_step': [35, 35, 36, 36], 'label': [1, 0, 1, 0]})
    out = utils.get_metrics_by_time_step(model, df, ['a'], 0.5)
    assert out['time_step'].tolist() == [35, 36]
    assert out['Fraud count'].tolist() == [1, 1]


def test_app_entrypoint_compiles():
    # main.py runs Streamlit code at import time, so only check that it parses
    import py_compile
    py_compile.compile(str(Path(__file__).resolve().parent.parent / 'app' / 'main.py'), doraise=True)
