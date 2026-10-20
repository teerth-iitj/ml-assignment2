import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

Q3_DIR = Path(__file__).resolve().parent
Q1_DIR = Q3_DIR.parent / "Q1"
sys.path.insert(0, str(Q1_DIR))

from bagging import Bagging
from common import load_data, get_seed

R, K, B = 30, 40, 100
DEPTHS = [4, 16, None]
DIS_REPS_NONE = 5
CACHE = Q3_DIR / "q3b_cache"
CACHE.mkdir(exist_ok=True)

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xte, yte = d["X_test"], d["y_test"]
SEED = get_seed()
RES = np.load(Q3_DIR / "q3_resamples.npy")
E = np.eye(K)[yte]


def full_proba(P, classes):
    out = np.zeros((P.shape[0], K)); out[:, classes] = P; return out


def leaf_stats(tree, Xr):
    is_leaf = tree.tree_.children_left == -1
    counts = np.bincount(tree.apply(Xr), minlength=tree.tree_.node_count)[is_leaf]
    return int(is_leaf.sum()), float(counts.mean()), float((counts == 1).mean())


def disagreement(bag, X):
    V = bag.vote_counts(X)
    agree = (V * (V - 1) / 2).sum(1)
    return float((1 - agree / (bag.B * (bag.B - 1) / 2)).mean())


for dep in DEPTHS:
    label = "None" if dep is None else str(dep)
    params = {"max_depth": dep, "min_samples_leaf": 1}
    for r in range(R):
        path = CACHE / f"d{label}_rep_{r:02d}.npz"
        if os.path.exists(path):
            continue
        idx = RES[r]; Xr, yr = Xtr[idx], ytr[idx]
        s = SEED + r + 1                       # same seeds as q3a.py
        tree = DecisionTreeClassifier(**params, random_state=s).fit(Xr, yr)
        nl, mpl, sf = leaf_stats(tree, Xr)
        Pt = full_proba(tree.predict_proba(Xte), tree.classes_)
        dis = float("nan")
        old = Q3_DIR / "q3a_cache" / f"rep_{r:02d}.npz"
        if dep is None:
            Pb = np.load(old)["P_bag"]                      # reuse the Q3(a) bagging fit
            if r < DIS_REPS_NONE:
                dis = disagreement(Bagging(B, params, seed=s).fit(Xr, yr), Xte)
        else:
            bag = Bagging(B, params, seed=s).fit(Xr, yr)
            Pb = bag.predict_proba(Xte)
            dis = disagreement(bag, Xte)
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, P_tree=Pt, P_bag=Pb, n_leaves=nl, mean_per_leaf=mpl,
                 frac_single=sf, dis=dis)
        os.replace(tmp, path)
        print(f"depth {label} rep {r+1}/{R} | tree acc {(Pt.argmax(1)==yte).mean():.3f} "
              f"bag acc {(Pb.argmax(1)==yte).mean():.3f}", flush=True)


def bv(P):
    pbar = P.mean(0)
    return float(((pbar - E) ** 2).sum(1).mean()), float(((P - pbar[None]) ** 2).sum(2).mean())


out = {}
print(f"\n{'depth':>5} | {'model':>7} {'bias2':>7} {'var':>7} {'acc':>15} | "
      f"{'leaves':>6} {'per_leaf':>8} {'single':>7} {'disagree':>8}")
for dep in DEPTHS:
    label = "None" if dep is None else str(dep)
    D = [np.load(CACHE / f"d{label}_rep_{r:02d}.npz") for r in range(R)]
    row = {}
    for name, key in [("tree", "P_tree"), ("bagging", "P_bag")]:
        P = np.stack([x[key] for x in D])
        b2, v = bv(P)
        acc = np.array([(p.argmax(1) == yte).mean() for p in P])
        row[name] = {"bias2": b2, "var": v, "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1))}
    row["leaves"] = float(np.mean([x["n_leaves"] for x in D]))
    row["per_leaf"] = float(np.mean([x["mean_per_leaf"] for x in D]))
    row["single"] = float(np.mean([x["frac_single"] for x in D]))
    row["disagree"] = float(np.nanmean([float(x["dis"]) for x in D]))
    row["var_removed"] = row["tree"]["var"] - row["bagging"]["var"]
    row["var_removed_frac"] = row["var_removed"] / row["tree"]["var"]
    row["acc_gain"] = row["bagging"]["acc_mean"] - row["tree"]["acc_mean"]
    out[label] = row
    for name in ["tree", "bagging"]:
        m = row[name]
        extra = (f"{row['leaves']:>6.1f} {row['per_leaf']:>8.2f} {row['single']:>7.3f} {row['disagree']:>8.3f}"
                 if name == "tree" else "")
        print(f"{label:>5} | {name:>7} {m['bias2']:>7.4f} {m['var']:>7.4f} "
              f"{m['acc_mean']:>7.3f} +- {m['acc_std']:.3f} | {extra}")
    print(f"{'':>5} | variance removed by bagging: {row['var_removed']:.4f} "
          f"({100*row['var_removed_frac']:.1f}%) | accuracy gain: {row['acc_gain']:+.3f}")

with (Q3_DIR / "q3b_results.json").open("w") as f:
    json.dump(out, f, indent=2)
print("\nSaved q3b_results.json")