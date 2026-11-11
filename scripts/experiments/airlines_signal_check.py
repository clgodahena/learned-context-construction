"""#5 — Airlines signal diagnostic (supervisor draft-2 feedback; extended for the
third review: three model families, so the result is not hostage to one learner).

This is a DIAGNOSTIC, not a proof of the dataset's signal ceiling: it reports
what three standard static learners (regularised logistic regression, a random
forest and histogram gradient boosting) reach under a temporal split, with naive
and with leak-safe categorical encoding, at the thesis cap and on a large sample.
A near-chance plateau across all of them is evidence that the stream carries
little static signal under these representations; it does not exclude that a
different representation or learner could do better.

Before concluding Airlines "has little usable signal", rule out that the near-
chance result is an artefact of (a) naive feature encoding and (b) the 5,700-
instance stream cap. This fits a strong STATIC learner under a temporal split
and reports balanced accuracy for the cross of:

    encoding ∈ {naive ordinal-as-numeric (what the stream pipeline does via
                cat.codes),  proper native-categorical handling}
    size     ∈ {5,700-row prefix (matches the thesis cap),  large sample}

Interpretation (printed at the end):
  * if proper+large lifts balanced accuracy well above ~0.55–0.60  → the
    "low signal" claim is partly our artefact; soften it and flag encoding.
  * if it stays near chance everywhere                            → the claim
    stands, now backed by evidence rather than reputation.

CPU only, sklearn + pandas; no TabPFN. Needs OpenML (cached locally once the
seed study has fetched Airlines). Run from the project root in the venv:
    python -m scripts.experiments.airlines_signal_check
"""
from __future__ import annotations
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    "rf":     lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                             class_weight="balanced_subsample", random_state=0),
    "hgb":    lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                     max_depth=None, random_state=0),
}

PREFIX = 5700           # the thesis stream cap
LARGE = 120_000         # large temporal sample to test the short-stream limit

def load():
    ds = fetch_openml(name="airlines", version=1, as_frame=True)
    f = ds.frame
    target = ds.target.name
    y = (f[target].astype(str).astype("category").cat.codes.to_numpy() if f[target].dtype.name in ("category", "object")
         else f[target].to_numpy()).astype(int)
    X = f.drop(columns=[target])
    # pandas dtype check: np.issubdtype cannot interpret OpenML's CategoricalDtype
    import pandas as pd
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols, target

def prep(X, cat_cols, num_cols):
    # Ordinal codes for categoricals (same raw representation the stream uses);
    # we then either HIDE their categorical nature (naive) or DECLARE it (proper).
    Xo = X.copy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xo[cat_cols] = enc.fit_transform(Xo[cat_cols].astype(str))
    for c in num_cols:
        Xo[c] = Xo[c].astype(float)
    return Xo.to_numpy(dtype=float), [Xo.columns.get_loc(c) for c in cat_cols]

def _encode_proper(Xtr, Xte, ytr, cat_idx):
    """Proper high-cardinality categorical handling: leak-safe target encoding
    (cross-fitted on the training fold; sklearn TargetEncoder). AirportFrom/To
    have >255 levels, beyond HGB's native-categorical limit, so target encoding
    is the standard treatment. Falls back to frequency encoding on old sklearn."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    try:
        from sklearn.preprocessing import TargetEncoder
        te = TargetEncoder(target_type="binary", random_state=0)
        Xtr[:, cat_idx] = te.fit_transform(Xtr[:, cat_idx], ytr)   # cross-fitted
        Xte[:, cat_idx] = te.transform(Xte[:, cat_idx])
        how = "target-encoded"
    except ImportError:
        for j in cat_idx:                                           # frequency encoding
            vals, cnt = np.unique(Xtr[:, j], return_counts=True)
            freq = dict(zip(vals, cnt / cnt.sum()))
            Xtr[:, j] = [freq.get(v, 0.0) for v in Xtr[:, j]]
            Xte[:, j] = [freq.get(v, 0.0) for v in Xte[:, j]]
        how = "frequency-encoded"
    return Xtr, Xte, how

def evaluate(Xall, yall, cat_idx, n, proper, model="hgb"):
    X, y = Xall[:n], yall[:n]
    cut = int(n * 0.8)                       # temporal split (respect arrival order)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    if proper and cat_idx:
        Xtr, Xte, _ = _encode_proper(Xtr, Xte, ytr, cat_idx)
    clf = MODELS[model]()
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    yhat = (p >= 0.5).astype(int)
    return (balanced_accuracy_score(yte, yhat),
            roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan"),
            average_precision_score(yte, p) if len(np.unique(yte)) > 1 else float("nan"),
            float(np.mean(y)))

def main():
    X, y, cat_cols, num_cols, target = load()
    N = len(y)
    print(f"Airlines: {N:,} rows, target={target!r}, pos_rate={y.mean():.3f}")
    print(f"categorical: {cat_cols}")
    print(f"numeric:     {num_cols}\n")
    Xo, cat_idx = prep(X, cat_cols, num_cols)
    large = min(LARGE, N)
    print(f"{'setting':34s} {'model':7s} {'bal_acc':>8} {'AUROC':>7} {'AUPRC':>7}")
    best = 0.0
    for n, label in [(PREFIX, f"{PREFIX}-row prefix (thesis cap)"), (large, f"{large:,}-row sample")]:
        for proper in (False, True):
            enc = "proper (target-encoded)" if proper else "naive ordinal-as-num"
            for model in ("logreg", "rf", "hgb"):
                ba, auc, ap, pr = evaluate(Xo, y, cat_idx, n, proper, model)
                best = max(best, ba)
                print(f"{label:>24s} | {enc:22s} {model:7s} {ba:8.4f} {auc:7.3f} {ap:7.3f}", flush=True)
    print(f"\nBest balanced accuracy over all settings and models: {best:.4f}")
    print("Reference: the TabPFN stream reports ~0.55-0.58 balanced accuracy on the 5,700 cap.")
    print("Read as a diagnostic: a plateau near 0.60 across three model families and two")
    print("encodings is evidence of little static signal under these representations, not a")
    print("proof of the dataset's signal ceiling.")

if __name__ == "__main__":
    main()
