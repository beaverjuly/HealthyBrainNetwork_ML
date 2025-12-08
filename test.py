# test.py
"""
Minimal tests for src/clusters_diagnosis.py

This script:
- checks that helper functions import correctly
- runs them on small synthetic data
- prints simple status messages

It does NOT import any real HBN data.
"""

import pandas as pd
import numpy as np

from src.clusters_diagnosis import (
    compute_cv_auc,
    cramers_v_bias_corrected,
    chi2_cramers_v,
    standardized_residuals_from_table,
    fdr_bh,
)


def test_compute_cv_auc():
    # tiny synthetic classification problem
    X = pd.DataFrame({
        "x1": [0, 1, 0, 1, 0, 1, 0, 1],
        "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    mean_auc, sd_auc = compute_cv_auc(X, y, n_splits=4, random_state=42)
    print(f"[OK] compute_cv_auc: mean={mean_auc:.3f}, sd={sd_auc:.3f}")


def test_cramers_and_residuals():
    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1, 2, 2, 3, 3],
        "dx":      [0, 1, 0, 1, 0, 0, 1, 1],
    })
    chi2, p, dof, V, tab = chi2_cramers_v(df["cluster"], df["dx"])
    print(f"[OK] chi2_cramers_v: chi2={chi2:.3f}, p={p:.3f}, V={V:.3f}, dof={dof}")

    resid_df = standardized_residuals_from_table(tab)
    print(f"[OK] standardized_residuals_from_table: shape={resid_df.shape}")


def test_fdr_bh():
    pvals = np.array([0.001, 0.02, 0.20, 0.90])
    rejected, p_adj = fdr_bh(pvals, alpha=0.05)
    print(f"[OK] fdr_bh:")
    print("     pvals   =", pvals)
    print("     rejected=", rejected)
    print("     p_adj   =", np.round(p_adj, 3))


def main():
    print("Running minimal tests for src/clusters_diagnosis.py ...")
    test_compute_cv_auc()
    test_cramers_and_residuals()
    test_fdr_bh()
    print("All tests ran without errors.")


if __name__ == "__main__":
    main()