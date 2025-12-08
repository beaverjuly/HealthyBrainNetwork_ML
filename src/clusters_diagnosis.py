from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def find_project_root(start: Path | None = None) -> Path:
    """Return the repo root by searching upward for markers."""
    p = (start or Path.cwd()).resolve()
    markers = {".git", "environment.yml", "README.md"}
    while True:
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            # fallback: use start if nothing found
            return (start or Path.cwd()).resolve()
        p = p.parent

def compute_cv_auc(X: pd.DataFrame,
                   y: pd.Series,
                   n_splits: int = 5,
                   random_state: int = 42):
    """
    Returns mean and SD of cross-validated AUC using
    LogisticRegression with standardization.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # helps with imbalance
                solver="lbfgs"
            )),
        ])

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        aucs.append(auc)

    aucs = np.array(aucs)
    return aucs.mean(), aucs.std()

def cramers_v_bias_corrected(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    # bias correction
    phi2_corr = max(0, phi2 - (k - 1)*(r - 1)/(n - 1))
    r_corr = r - (r - 1)**2 / (n - 1)
    k_corr = k - (k - 1)**2 / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

def chi2_cramers_v(x: pd.Series, y: pd.Series):
    """Convenience wrapper: returns chi2, p, dof, Cramér's V (bias-corrected)."""
    tab = pd.crosstab(x, y)
    chi2, p, dof, _ = chi2_contingency(tab)
    V = cramers_v_bias_corrected(tab)
    return chi2, p, dof, V, tab

def standardized_residuals_from_table(obs: pd.DataFrame) -> pd.DataFrame:
    chi2, p, dof, expected = chi2_contingency(obs.values)
    resid = (obs.values - expected) / np.sqrt(expected)
    return pd.DataFrame(resid, index=obs.index, columns=obs.columns)

def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    rejected = np.zeros_like(pvals, dtype=bool)
    if below.any():
        k_max = np.where(below)[0].max()
        cutoff = ranked[k_max]
        rejected = pvals <= cutoff

    # adjusted p-values
    p_adj = np.empty_like(pvals, dtype=float)
    p_adj[order] = np.minimum.accumulate((n / np.arange(1, n + 1)) * ranked[::-1])[::-1]
    p_adj = np.minimum(p_adj, 1.0)
    return rejected, p_adj