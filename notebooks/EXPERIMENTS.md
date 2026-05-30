# Experiments — ideas to try

Side experiments worth running, with hypothesis + success criterion so results are judged, not just produced.

## PCA → LogReg (dimensionality reduction)

**Idea.** Apply PCA to the 15-feature contract, then fit Logistic Regression on the components; compare to the no-PCA baseline.

**Hypothesis (expected to NOT help here):**
- 15 features is already low-dimensional — little to reduce.
- Tree models (XGB/LGBM/Cat/RF) are scale-invariant and handle correlated/redundant features natively, so PCA's rotated combos usually give *worse* splits → run PCA only with **LogReg**, not trees.
- PCA is unsupervised (maximizes variance, not target separation) — may down-weight `EXT_SOURCE_*` (strongest signal).
- **Biggest cost: interpretability.** Components are opaque linear blends → breaks per-feature "Top drivers" attribution and credit governance. Not servable to the frontend.

**Setup.**
- Pipeline: `SimpleImputer` → `StandardScaler` → `PCA(n_components=…)` → `LogisticRegression(C, penalty='l2')`.
- Data: `build_model_matrix(app, bureau, prev)` (15 features).
- CV: `StratifiedKFold(5)`, metric ROC AUC (+ KS).
- Sweep `n_components` ∈ {5, 8, 10, 12, 15}; also try `explained_variance` ≥ 0.95.

**Baselines to beat.**
- LogReg on the 15 features **without** PCA (L2-regularized).
- Best tree model (LightGBM) — PCA-LogReg will almost certainly lose to this.

**Success criterion.** PCA-LogReg AUC ≥ no-PCA LogReg AUC by a meaningful margin (> ~0.003) on OOF. If not (expected), record it as a negative result and keep PCA out of the production pipeline.

**Decision rule.** Even if AUC ties, **do not** use PCA in serving — it kills the per-feature explanation the UI and model card require. PCA stays a learning experiment only.

**Status:** ✅ DONE — see `notebooks/exp_pca.ipynb`. **Result: PCA does not help.**
- LogReg (no PCA) OOF AUC **0.7279** / KS 0.335.
- `PCA(n=15)` = 0.7279 (full rotation is lossless); fewer components strictly worse:
  n=12 −0.0016, n=10 −0.0045, n=8 −0.0119, n=5 −0.0212.
- Explained variance is gradual (≈12 comps for 95%, no elbow) → features aren't redundant; PCA can't
  compress without dropping signal.
- **Conclusion:** keep PCA out of the production pipeline (no AUC gain + loses per-feature interpretability).
