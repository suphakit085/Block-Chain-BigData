"""
Training recipe shared by scripts/train_models.py and the dashboard.
The dashboard calls train_models() when no saved models exist (e.g. on
Streamlit Community Cloud), so the hosted app needs no pickles in the repo.
"""
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TRAIN_END = 34  # time steps 1-34 train, 35-49 are held out for the dashboard
NON_FEATURES = ['label', 'txId', 'time_step', 'class']


def model_filename(name):
    return f"{name.replace(' ', '_').lower()}_smote_v1_graph.pkl"


def make_models():
    return {
        # Unscaled LR is unusable on this data (test PR-AUC ~0.19); scaling + stronger L2 -> ~0.36
        'Logistic Regression': make_pipeline(
            StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    }


def train_models(df, names=None, log=print):
    """Fit the models on time steps 1-34 (SMOTE on train only). `df` must have a time_step column."""
    train = df[df['time_step'] <= TRAIN_END]
    X = train.drop(columns=NON_FEATURES, errors='ignore')
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, train['label'])
    log(f"Train rows: {len(X):,} -> {len(X_res):,} after SMOTE")

    fitted = {}
    for name, model in make_models().items():
        if names is not None and name not in names:
            continue
        log(f"Training {name}...")
        fitted[name] = model.fit(X_res, y_res)
    return fitted
