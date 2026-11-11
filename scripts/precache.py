"""Warm every network-dependent cache BEFORE travelling.

Run this once on a stable internet connection. After it finishes, the seed
study needs no network at all — so the Mac moving between home and office (and
the internet dropping on the way) cannot interrupt a run.

It fetches and caches:
  * Electricity (OpenML)      -> ~/scikit_learn_data
  * Airlines (OpenML)         -> ~/scikit_learn_data
  * Covertype (scikit-learn)  -> ~/scikit_learn_data
  * TabPFN weights            -> downloaded + one tiny prediction to force load
                                 (also lets you accept the TabPFN licence once,
                                 interactively, before any batch run)

Usage:
  python -m scripts.precache
"""
from __future__ import annotations

import time


def _t(label, fn):
    t0 = time.time()
    try:
        fn()
        print(f"  [ok]   {label}  ({time.time()-t0:.1f}s)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {type(e).__name__}: {str(e)[:200]}")
        return False


def cache_electricity():
    from src.data.streams import electricity
    s = electricity()
    assert s.n_features and s.n_classes


def cache_airlines():
    from src.data.streams import airlines
    s = airlines()
    assert s.n_features and s.n_classes


def cache_covertype():
    from src.data.streams import covertype
    s = covertype()
    assert s.n_features and s.n_classes


def warm_tabpfn():
    """Build TabPFN and run one tiny prediction so weights download + load now."""
    import numpy as np
    from src.classifiers.tabpfn_clf import TabPFNInContextClassifier
    clf = TabPFNInContextClassifier(n_classes=2, device=None, max_context=256)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 8)).astype("float32")
    y = (rng.random(32) > 0.5).astype(int)
    proba = clf.predict_proba(X, y, rng.normal(size=8).astype("float32"))
    assert proba is not None
    print(f"         TabPFN device = {clf._resolve_device()}")


def main():
    print("[precache] warming caches (run this on a stable connection)...")
    results = {
        "Electricity (OpenML)": _t("Electricity (OpenML)", cache_electricity),
        "Airlines (OpenML)": _t("Airlines (OpenML)", cache_airlines),
        "Covertype (scikit-learn)": _t("Covertype (scikit-learn)", cache_covertype),
        "TabPFN weights + load": _t("TabPFN weights + load", warm_tabpfn),
    }
    ok = sum(results.values())
    print(f"\n[precache] {ok}/{len(results)} caches ready.")
    if ok == len(results):
        print("[precache] All set — you can now run the study offline, moving the Mac freely.")
    else:
        print("[precache] Some caches failed. Re-run on a good connection before travelling;\n"
              "           a failed Covertype/Airlines fetch will otherwise stall those trials.")


if __name__ == "__main__":
    main()
