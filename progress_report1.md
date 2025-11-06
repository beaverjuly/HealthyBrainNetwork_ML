**Student:** Julia Yi  
**Repo:** https://github.com/beaverjuly/hbn_project  
**Environment:** conda (`hbn-ml`) via `environment.yml`  
**Notebook(s):** `notebooks/01_ingest_qc.ipynb`

## 1) Scope / Goal
Unsupervised learning on Healthy Brain Network (HBN) public phenotypes to explore transdiagnostic subgroup structure, then expand with symptom/cognition totals upon DUA approval.

## 2) Data sources (≥1 via API/web)
- **HBN Basic Phenotype (Release 11)** — CSV fetched programmatically  
  Env var: `HBN_PUBLIC_CSV_URL`  
  Example: http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R11_Pheno.csv  
  Access: `requests` + `pandas` (`src/io_utils.py`, `tests.py`)  
  Current shape: **1160 × 6**.

(Planned) Add a second public HBN CSV and, when approved, NIH Toolbox/questionnaire totals via DUA.

## 3) Pipeline (current status)
A → B → C → F → G → H-lite → K  
- **Ingest (A):** programmatic CSV fetch; saved to `data/` (gitignored).  
- **QC (B):** type coercion, median imputation, 1–99% clipping.  
- **Standardization (C):** z-scores.  
- **Clustering (F):** KMeans (k=3–4, `n_init=20`); Agglomerative (Ward on PCA) planned.  
- **Selection (G):** silhouette done; CH/DB next; ≤100 bootstrap resamples.  
- **Interpretation (H-lite):** cluster centroids (z) and counts.  
- **Reporting (K):** heatmap, PCA/UMAP scatter, bar charts; plain-language summary.

## 4) Results so far
- API test passes (`python tests.py` → OK: (1160, 6)).  
- First KMeans run completed; silhouette = **[fill from notebook]**.  
- Initial centroids indicate separations along available public features (to be refined with symptom/cognition totals).

## 5) Risks / Issues
- DUA timing; mitigation: proceed with public phenotypes, slot in totals later with no pipeline changes.

## 6) Next 7–10 days
- Add second public CSV and merge on participant ID.  
- Compute CH/DB and bootstrap stability.  
- If DUA approved: integrate 2–3 symptom totals + 3 NIH Toolbox scores (+ age/sex).  
- Produce final plots and update report.