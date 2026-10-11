"""Does a selected subset actually train a better model? A downstream test of the curation samplers.

Protocol: split the images 70/30 (grouped by video). A sampler picks `budget` of the 70% pool using one
embedding model (the selection model). A logistic-regression probe is trained on the selected images'
features from the OTHER embedding model, so the probe does not see the space the sampler used, and is scored
on the untouched 30% for each label tag. Reported: macro-F1 and recall on rare classes (classes that make up
at most `rare_frac` of the pool; unlabelled values excluded). `all_data` trains on the whole pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from dgeo.curation.metrics import UNLABELLED, rare_values
from dgeo.curation.samplers import SAMPLERS


def probe_scores(Xtr, ytr, Xte, yte, rare: set) -> dict:
    classes = np.unique(ytr)
    if len(classes) < 2:
        return dict(macro_f1=np.nan, rare_recall=np.nan, rare_classes_seen=0, n_rare_test=0)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    keep = ~np.isin(yte, list(UNLABELLED))
    labels = list(np.unique(yte[keep]))
    mf1 = f1_score(yte[keep], pred[keep], average="macro", labels=labels, zero_division=0)
    rare_mask = np.isin(yte, list(rare))
    rr = float((pred[rare_mask] == yte[rare_mask]).mean()) if rare_mask.sum() else np.nan
    return dict(macro_f1=float(mf1), rare_recall=rr, rare_classes_seen=int(len(rare & set(classes))),
                n_rare_test=int(rare_mask.sum()))


def run(embs: dict, manifest: pd.DataFrame, tags: dict, budget: float, samplers: list, n_splits: int, seeds: int,
        rare_frac: float = 0.05) -> pd.DataFrame:
    groups = manifest["video"].astype(str).values
    names = list(embs)
    rows = []
    for split in range(n_splits):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=split).split(np.zeros(len(groups)),
                                                                                    groups=groups))
        k = max(2, int(round(budget * len(tr))))
        for sel_model in names:
            probe_models = [m for m in names if m != sel_model] + [sel_model]
            for sname in samplers + ["all_data"]:
                n_seeds = 1 if sname in ("proto_hard", "all_data") else seeds
                for seed in range(n_seeds):
                    sel = tr if sname == "all_data" else tr[SAMPLERS[sname](embs[sel_model][tr], k, seed=seed)]
                    for pm in probe_models:
                        for tname, tarr in tags.items():
                            rare = set(rare_values(tarr[tr], rare_frac))
                            sc = probe_scores(embs[pm][sel], tarr[sel], embs[pm][te], tarr[te], rare)
                            rows.append(dict(split=split, seed=seed, select_model=sel_model, probe_model=pm,
                                             same_model=pm == sel_model, sampler=sname, tag=tname, k=len(sel), **sc))
        print(f"[downstream] split {split} done", flush=True)
    return pd.DataFrame(rows)
