# Data-Driven EF/Temporal Discounting Profiles and Diagnosis in the Healthy Brain Network

## Project question

**Central question**

> How do data-driven summaries of cognitive control and reward discounting relate to neurodevelopmental and anxiety diagnoses in youth?

The goal is not to build a better diagnostic classifier, but to see whether a small number of EF/temporal-discounting (TD) profiles can help **organize heterogeneity and comorbidity** in these diagnoses.

---

## Data

All data come from the **Healthy Brain Network (HBN)**. Raw files are not stored in this repo; the pipeline assumes local CSVs under `data/`.

Key ingredients:

- **Executive function (EF)**  
  Age-corrected NIH Toolbox scores (e.g., Flanker, List Sorting, Processing Speed).

- **Temporal discounting (TD)**  
  Subject-level discounting parameters from a money choice task (smaller-sooner vs larger-later), summarized as:
  - `logk_mean_z`: standardized mean log-discount rate (higher = stronger “now bias”).  
  - `logk_diff_z`: standardized run-to-run difference (variability in discounting).

- **Diagnosis flags**  
  Clinician-consensus binary indicators for:
  - Any neurodevelopmental disorder (ND)  
  - Specific learning disorder (SLD)  
  - Autism spectrum disorder (ASD)  
  - ADHD  
  - Anxiety disorder  

- **Processed analysis table**  
  A merged, cleaned table joining EF/TD features, demographics, KMeans cluster labels, and diagnosis flags
  (e.g., `data/processed/hbn_core_clusters_diag.csv`).

---

## Analysis overview

The full workflow is spread across several notebooks; the main *results* are consolidated in `results.ipynb`.

### 1. Preprocessing (supporting notebooks)

- **`01_...` / `05_...`**  
  - Pull public phenotype via HTTP (HBN pheno CSV).  
  - Merge NIH Toolbox EF, TD parameters, and diagnosis flags.  
  - Clean and standardize EF/TD features; construct `logk_mean_z` and `logk_diff_z`.  
  - Save core processed views used downstream.

### 2. Clustering & method selection (supporting notebooks)

- **`03_clustering_method_selection.ipynb`**  
  - Work in EF/TD feature space only (no diagnoses).  
  - Compare multiple clustering methods / hyperparameters.  
  - Use internal metrics (e.g., silhouette × coverage, stability, EF/TD separation) to choose a **4-cluster KMeans** solution as a good balance of separation, stability, and interpretability.

- **`04_KMeans.ipynb`**  
  - Fit the final **4-cluster KMeans** model on EF/TD features.  
  - Save subject-level cluster assignments to `results/kmeans_model/cluster_assignments.csv`.

### 3. Cluster–diagnosis analyses (main results notebook)

- **`results.ipynb`** (final, graded notebook)
  - Uses helper functions from `src/clusters_diagnosis.py`.
  - Steps:
    1. **Load**:
       - processed EF/TD table (`hbn_core_view_v1.csv` or similar),
       - saved KMeans cluster labels (`cluster_assignments.csv`),
       - diagnosis flags (`hbn_diag_flags_neuro_anx.csv`).
    2. **Merge** into a single analysis table with EF/TD, cluster labels, and diagnosis flags.
    3. **Describe cluster profiles** (size, age/sex, mean EF/TD z-scores) and visualize as a cluster-by-feature heatmap.
    4. **Cluster–diagnosis association tests**  
       - For each diagnosis (Any ND, SLD, ASD, ADHD, Anxiety):
         - Build a 4×2 contingency table: clusters × yes/no diagnosis.
         - Compute χ² and **bias-corrected Cramér’s V** (effect size) using `chi2_cramers_v` and `cramers_v_bias_corrected`.
         - Compute standardized residuals for cell-wise over-/under-representation.
         - Apply **Benjamini–Hochberg FDR correction** across diagnoses with `fdr_bh`.
    5. **Diagnostic utility (AUC)**  
       - For each diagnosis, fit logistic regression with cross-validated AUC via `compute_cv_auc`:
         - EF/TD features only.  
         - EF/TD + cluster membership (one-hot).  
         - Clusters only.  
       - Summarize and plot AUCs and ΔAUC (incremental value of clusters beyond EF/TD).

---

## Main findings

- The 4-cluster KMeans solution yields **interpretable EF/TD profiles**, e.g.:
  - relatively high EF and more patient discounting,
  - lower EF with stronger “now bias,”
  - lower EF but relatively patient,
  - more variable discounting.

- **Associations with diagnoses** (Cramér’s V):
  - Small but non-zero links with **Any ND** and **SLD**, weaker or negligible for ASD, ADHD, and Anxiety after FDR correction.
  - Indicates substantial **within-diagnosis heterogeneity** in EF/TD profiles.

- **Predictive (diagnostic) utility**:
  - Adding clusters to EF/TD features yields **essentially no gain in AUC**.
  - Clusters alone perform slightly worse than EF/TD alone.
  - Clusters are therefore **not useful as standalone diagnostic predictors** in this dataset.

- **Interpretation**:
  - Clusters are better viewed as **descriptive and interpretive tools**:
    - They compress EF/TD space into a small number of cognitive–motivational profiles that are easier to discuss.
    - They help describe and visualize heterogeneity and comorbidity across neurodevelopmental and anxiety diagnoses, rather than replace diagnoses.

---

## Code structure

- **`results.ipynb`**  
  Main notebook for the final analysis:
  - Loads processed EF/TD + cluster + diagnosis data.
  - Calls helper functions from `src/clusters_diagnosis.py`.
  - Generates all cluster–diagnosis statistics and plots.

- **`src/clusters_diagnosis.py`**  
  Helper module providing:
  - `find_project_root`: locate the project root directory.  
  - `compute_cv_auc`: cross-validated AUC for logistic regression with standardized features.  
  - `cramers_v_bias_corrected`: bias-corrected Cramér’s V from a contingency table.  
  - `chi2_cramers_v`: wrapper that returns χ², p, dof, Cramér’s V, and the contingency table.  
  - `standardized_residuals_from_table`: standardized residuals from χ² expected vs observed counts.  
  - `fdr_bh`: Benjamini–Hochberg FDR correction on a vector of p-values.

- **Supporting notebooks**  
  - `01_...` / `05_...`: data fetching, cleaning, and EF/TD feature construction.  
  - `03_clustering_method_selection.ipynb`: internal clustering method comparison.  
  - `04_KMeans.ipynb`: final KMeans fit and export of cluster assignments.

## Tests

The repository includes a small `test.py` script that runs smoke tests on the
helper functions in `src/clusters_diagnosis.py` using synthetic data
(no HBN data required).

To run the tests (after creating and activating the environment and installing
dependencies):

```bash
# 1. Clone and move into the project directory
git clone https://github.com/beaverjuly/HealthyBrainNetwork_ML.git
cd HealthyBrainNetwork_ML

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate hbn-ml

# 3. Run minimal smoke tests for the helper module
python test.py
```