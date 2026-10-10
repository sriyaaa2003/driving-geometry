import numpy as np

from dgeo.curation.metrics import linear_cka, tag_metrics
from dgeo.curation.samplers import SAMPLERS, water_fill
from dgeo.retrieval import knn_cosine, recall_at_k
from tests.fixtures import make_synthetic_embeddings, make_synthetic_vpr


def test_water_fill_respects_caps_and_budget():
    rng = np.random.default_rng(0)
    caps = np.array([1, 50, 3, 100])
    a = water_fill(40, caps, np.ones(4), rng)
    assert a.sum() == 40 and (a <= caps).all()
    assert a[0] == 1 and a[2] == 3


def test_all_samplers_return_k_unique_indices():
    emb, _ = make_synthetic_embeddings(n=1500, d=32, seed=1)
    for name, f in SAMPLERS.items():
        sel = f(emb, 100, seed=0)
        assert len(sel) == 100, name
        assert len(np.unique(sel)) == 100, name
        assert sel.min() >= 0 and sel.max() < len(emb), name


def test_balanced_hierarchy_covers_rare_scenes_better_than_random():
    emb, tags = make_synthetic_embeddings(n=4000, d=48, seed=2)
    k = 120
    rl = {}
    for name in ["random", "hier_balanced"]:
        vals = [tag_metrics(SAMPLERS[name](emb, k, seed=s), tags["scenario"], 0.05)["rare_lift"]
                for s in range(4)]
        rl[name] = float(np.mean(vals))
    assert rl["hier_balanced"] > 1.5 * rl["random"], rl


def test_cka_identity_and_orthogonal_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 20))
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert abs(linear_cka(x, x) - 1) < 1e-9
    assert abs(linear_cka(x, x @ q) - 1) < 1e-9


def test_vpr_recall_full_db_is_high_without_shift():
    db, db_xy, q, q_xy = make_synthetic_vpr(shift=0.0, seed=3)
    idx, _ = knn_cosine(q, db, k=5)
    assert recall_at_k(q_xy, db_xy, idx, thr_m=60.0)["R@5"] > 0.5
