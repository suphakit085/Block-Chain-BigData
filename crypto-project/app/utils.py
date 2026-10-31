import pandas as pd
import pickle
import os
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, precision_recall_curve
)
import numpy as np

# Constants
from pathlib import Path

# Resolve paths from the repo root so the app works regardless of the launch directory
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv')
MODELS_DIR = str(ROOT / 'models')
# Compact, lossless copies used when the full CSVs are absent (hosted deployment)
DEPLOY_DIR = ROOT / 'deploy_data'
DEPLOY_DATA_PATH = str(DEPLOY_DIR / 'elliptic_v1_graph.parquet')
DEPLOY_EDGELIST_PATH = str(DEPLOY_DIR / 'elliptic_txs_edgelist.parquet')


@st.cache_data
def load_data():
    """Loads the V1 graph dataset: the local CSV if present, else the bundled parquet."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        elif os.path.exists(DEPLOY_DATA_PATH):
            df = pd.read_parquet(DEPLOY_DATA_PATH)
        else:
            st.error(f"Dataset not found at {DATA_PATH} or {DEPLOY_DATA_PATH}")
            return None
        if 'feat_0' in df.columns:
            df = df.rename(columns={'feat_0': 'time_step'})
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Loads the saved V1 models from models/. Any that are missing are trained from
    the loaded data (same recipe as scripts/train_models.py) and saved when possible,
    so a hosted deployment works without pickles in the repo.
    """
    from training import make_models, model_filename, train_models

    models, missing = {}, []
    for name in make_models():
        path = os.path.join(MODELS_DIR, model_filename(name))
        if os.path.exists(path):
            with open(path, 'rb') as file:
                models[name] = pickle.load(file)
        else:
            missing.append(name)

    if missing:
        df = load_data()
        if df is None:
            return models
        with st.spinner(f"Training {', '.join(missing)} (first start only, ~1 min)..."):
            trained = train_models(df, names=missing, log=lambda *_: None)
        for name, model in trained.items():
            models[name] = model
            try:
                os.makedirs(MODELS_DIR, exist_ok=True)
                with open(os.path.join(MODELS_DIR, model_filename(name)), 'wb') as file:
                    pickle.dump(model, file)
            except OSError:
                pass  # read-only filesystem: keep the in-memory model only
    return dict(sorted(models.items()))  # stable order: LR, RF, XGBoost

def _unwrap(model, X):
    """
    Split a sklearn Pipeline into (final_estimator, X transformed by the earlier steps).
    Plain estimators are returned unchanged. Explainers and importances work on the
    final estimator, in the feature space it was fitted on (e.g. standardised for LR).
    """
    if hasattr(model, 'steps') and len(model.steps) > 1:
        Xt = model[:-1].transform(X)
        return model[-1], pd.DataFrame(Xt, columns=X.columns, index=X.index)
    return model, X


def get_feature_importance(model, feature_names):
    """Extracts feature importance if the model supports it."""
    model = model[-1] if hasattr(model, 'steps') else model
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For Logistic Regression (using absolute coefficients)
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(model.coef_[0])
        }).sort_values(by='Importance', ascending=False)
    return None


def predict_with_threshold(model, X, threshold=0.5):
    """Flag a transaction as fraud when P(fraud) >= threshold."""
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def get_model_metrics(models, X_test, y_test, threshold=0.5):
    """
    Calculate performance metrics for all models at the given alert threshold.
    Returns a DataFrame with Accuracy, Precision, Recall, F1 and PR-AUC for each model.
    PR-AUC (average precision) does not depend on the threshold and is the
    fairest single number for an imbalanced problem like AML.
    """
    results = []

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'PR-AUC': average_precision_score(y_test, y_prob)
        })

    return pd.DataFrame(results)


def get_confusion_matrix_data(model, X_test, y_test, threshold=0.5):
    """
    Generate confusion matrix data for a given model at the given threshold.
    Returns the confusion matrix as a numpy array.
    """
    y_pred = predict_with_threshold(model, X_test, threshold)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return cm


def get_pr_curve(model, X_test, y_test):
    """Precision-recall curve points for the selected model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    # precision/recall have one more point than thresholds; drop the last one
    return pd.DataFrame({
        'Threshold': thresholds,
        'Precision': precision[:-1],
        'Recall': recall[:-1],
    })


def get_precision_at_k(model, X_test, y_test, ks=(100, 250, 500, 1000)):
    """
    "If the compliance team can review only K alerts, how many are real fraud?"
    Ranks test transactions by fraud probability and checks the top K.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_test)[order]
    total_fraud = int(y_sorted.sum())
    rows = []
    for k in ks:
        if k > len(y_sorted):
            continue
        hits = int(y_sorted[:k].sum())
        rows.append({
            'Review budget (K)': k,
            'Fraud found': hits,
            'Precision@K': hits / k,
            'Share of all fraud caught': hits / total_fraud if total_fraud else 0.0,
        })
    return pd.DataFrame(rows)


def get_metrics_by_time_step(model, test_df, feature_cols, threshold=0.5):
    """
    Per-time-step precision/recall on the test period.
    Shows whether the model degrades over time (concept drift).
    """
    rows = []
    for t, group in test_df.groupby('time_step'):
        y_true = group['label']
        y_pred = predict_with_threshold(model, group[feature_cols], threshold)
        rows.append({
            'time_step': int(t),
            'Fraud count': int(y_true.sum()),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


DRIFT_STEP = 43  # dark-market shutdown in the Elliptic data; patterns change from here


def get_metrics_by_period(model, test_df, feature_cols, threshold=0.5, split_step=DRIFT_STEP):
    """
    Precision/recall/F1 before and after `split_step`.
    The overall test-set F1 hides that recall collapses after the regime change.
    """
    rows = []
    for name, part in [(f"Steps {int(test_df['time_step'].min())}-{split_step - 1}", test_df[test_df['time_step'] < split_step]),
                       (f"Steps {split_step}-{int(test_df['time_step'].max())}", test_df[test_df['time_step'] >= split_step])]:
        if part.empty:
            continue
        y_true = part['label']
        y_pred = predict_with_threshold(model, part[feature_cols], threshold)
        rows.append({
            'Period': name,
            'Illicit': int(y_true.sum()),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


# ============================================
# Transaction Network (graph) Functions
# ============================================
EDGELIST_PATH = str(ROOT / 'data/raw/elliptic_txs_edgelist.csv')


@st.cache_resource
def load_transaction_graph():
    """
    Builds the directed Bitcoin transaction graph (txId1 -> txId2) once and caches it.
    Returns None if the edge list file is not available.
    """
    import networkx as nx
    if os.path.exists(EDGELIST_PATH):
        edges = pd.read_csv(EDGELIST_PATH)
    elif os.path.exists(DEPLOY_EDGELIST_PATH):
        edges = pd.read_parquet(DEPLOY_EDGELIST_PATH)
    else:
        return None
    return nx.from_pandas_edgelist(
        edges, source=edges.columns[0], target=edges.columns[1], create_using=nx.DiGraph()
    )


def collect_neighborhood(G, tx_id, label_map, hops=1, max_nodes=150, explore_limit=20000):
    """
    Breadth-first neighbourhood of `tx_id` (money in or out), nearest hops first.

    Whole hop levels are kept while they fit in `max_nodes`; in the level that
    overflows, known-illicit transactions are kept first, then licit, then unknown.
    Nearer transactions are therefore never dropped in favour of farther ones.

    Returns a dict with the kept node set and what was found before truncation:
      nodes, distance (node -> hops), found (total within `hops`), illicit_found,
      truncated (bool), complete (False if exploration stopped at `explore_limit`).
    """
    distance = {tx_id: 0}
    frontier = [tx_id]
    complete = True
    for d in range(1, hops + 1):
        nxt = set()
        for n in frontier:
            nxt.update(G.successors(n))
            nxt.update(G.predecessors(n))
        nxt = [n for n in nxt if n not in distance]
        for n in nxt:
            distance[n] = d
        frontier = nxt
        if not frontier:
            break
        if len(distance) >= explore_limit:
            complete = False
            break

    priority = {1: 0, 0: 1}  # illicit, then licit, then unknown
    ranked = sorted(distance, key=lambda n: (distance[n], priority.get(label_map.get(n, -1), 2), n))
    kept = set(ranked[:max_nodes])
    illicit_found = sum(1 for n in distance if n != tx_id and label_map.get(n, -1) == 1)
    return {
        'nodes': kept,
        'distance': distance,
        'found': len(distance),
        'illicit_found': illicit_found,
        'truncated': len(distance) > len(kept),
        'complete': complete,
    }


def build_neighbor_graph_html(G, tx_id, label_map, hops=1, max_nodes=150):
    """
    Renders the neighbourhood of one transaction as interactive HTML (pyvis).
    Red = illicit, green = licit, grey = unknown label.
    Returns (html, info) where info is collect_neighborhood()'s summary plus
    'shown' and 'illicit_shown'; (None, None) if tx is not in the graph.
    """
    from pyvis.network import Network

    if G is None or tx_id not in G:
        return None, None

    info = collect_neighborhood(G, tx_id, label_map, hops=hops, max_nodes=max_nodes)
    nodes = info['nodes']

    sub = G.subgraph(nodes)
    net = Network(height='420px', width='100%', bgcolor='#0E1117', font_color='#FAFAFA',
                  directed=True, cdn_resources='in_line')  # embed JS so it works inside Streamlit/offline
    net.barnes_hut()

    n_illicit = 0
    for n in sub.nodes():
        label = label_map.get(n, -1)
        if n != tx_id and label == 1:
            n_illicit += 1
        color = '#FF2D7C' if label == 1 else ('#00FF88' if label == 0 else '#666666')
        status = 'Illicit' if label == 1 else ('Licit' if label == 0 else 'Unknown')
        net.add_node(
            str(n),
            label='THIS TX' if n == tx_id else str(n)[-4:],
            title=f"txId {n} — {status} — {info['distance'][n]} hop(s) away",
            color={'background': color, 'border': '#00D4FF' if n == tx_id else color},
            size=28 if n == tx_id else 12,
            borderWidth=4 if n == tx_id else 1,
        )
    for u, v in sub.edges():
        net.add_edge(str(u), str(v), color='rgba(255,255,255,0.35)', arrows='to')

    info['shown'] = sub.number_of_nodes()
    info['illicit_shown'] = n_illicit
    return net.generate_html(notebook=False), info


def create_export_data(tx_data, prediction, probability, model_name):
    """
    Create a DataFrame for exporting prediction results.
    """
    export_df = tx_data.copy()
    export_df['Prediction'] = 'Fraud' if prediction == 1 else 'Safe'
    export_df['Fraud_Probability'] = probability
    export_df['Model_Used'] = model_name
    return export_df


# ============================================
# SHAP Analysis Functions
# ============================================
import shap


def _fraud_class_shap(shap_values):
    """
    Return SHAP values for the fraud class as a 2D (rows, features) array.
    Older shap returns a list [class0, class1]; newer versions return a 3D
    array (rows, features, classes) for sklearn tree ensembles.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


@st.cache_data
def get_shap_global_values(_model, X_sample, _model_name):
    """
    Calculate SHAP values for global feature importance.
    Uses a sample of 500 rows for speed.
    _model and _model_name are unhashable-prefixed to allow caching.
    """
    try:
        # Sample background data for speed
        if len(X_sample) > 500:
            X_bg = X_sample.sample(500, random_state=42)
        else:
            X_bg = X_sample
        
        _model, X_bg = _unwrap(_model, X_bg)
        
        # Choose explainer based on model type
        if hasattr(_model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost)
            explainer = shap.TreeExplainer(_model)
            shap_values = _fraud_class_shap(explainer.shap_values(X_bg))
        elif hasattr(_model, 'coef_'):
            # Linear models (Logistic Regression)
            explainer = shap.LinearExplainer(_model, X_bg)
            shap_values = explainer.shap_values(X_bg)
        else:
            return None, None
        
        # Calculate mean absolute SHAP values per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': X_bg.columns.tolist(),
            'SHAP_Importance': mean_shap
        }).sort_values('SHAP_Importance', ascending=False)
        
        return feature_importance, None
    except Exception as e:
        st.warning(f"SHAP calculation error: {e}")
        return None, None


def get_shap_single_values(model, features, X_background, model_name):
    """
    Calculate SHAP values for a single transaction prediction.
    Returns shap_values array and base_value for waterfall plot.
    """
    try:
        # Sample background data
        if len(X_background) > 100:
            X_bg = X_background.sample(100, random_state=42)
        else:
            X_bg = X_background
        
        _, features = _unwrap(model, features)
        model, X_bg = _unwrap(model, X_bg)
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = _fraud_class_shap(explainer.shap_values(features))
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)) and np.size(base_value) > 1:
                base_value = base_value[1]
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_bg)
            shap_values = explainer.shap_values(features)
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = np.ravel(base_value)[0]
        else:
            return None, None
        
        return shap_values[0], base_value
    except Exception as e:
        st.warning(f"SHAP single prediction error: {e}")
        return None, None
