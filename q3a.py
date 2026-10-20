import json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

Q3_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = Q3_DIR.parent
Q1_DIR = ASSIGNMENT_DIR / "Q1"
Q2_DIR = ASSIGNMENT_DIR / "Q2"
sys.path.insert(0, str(Q2_DIR))
sys.path.insert(0, str(Q1_DIR))

from samme import SAMME
from bagging import Bagging
from common import load_data, get_seed

R, K = 30, 40
CACHE = Q3_DIR / "q3a_cache"
CACHE.mkdir(exist_ok=True)

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xte, yte = d["X_test"], d["y_test"]
Xtm = d["X_test_masked"]
SEED = get_seed()

with (Q1_DIR / "frozen_tree.json").open() as f:
    tp = json.load(f)
with (Q1_DIR / "final_B.json").open() as f:
    B = json.load(f)["B"]
with (Q2_DIR / "boost_final.json").open() as f:
    bf = json.load(f)
tree_params = {"max_depth": tp["max_depth"], "min_samples_leaf": tp["min_samples_leaf"]}

# ---- the 30 resampled training sets (created once, then reused by q3b.py) ----
resamples_path = Q3_DIR / "q3_resamples.npy"
if resamples_path.exists():
    RES = np.load(resamples_path)
else:
    rng = np.random.default_rng([SEED, 3])
    RES = np.zeros((R, len(ytr)), dtype=int)
    for r in range(R):
        parts = []
        for k in range(K):
            pos = np.where(ytr == k)[0]                  # the identity's 6 original training faces
            parts.append(rng.choice(pos, size=len(pos), replace=True))
        RES[r] = np.concatenate(parts)
    np.save(resamples_path, RES)
assert RES.shape == (R, 240)


def full_proba(P, classes):
    out = np.zeros((P.shape[0], K))
    out[:, classes] = P
    return out


E = np.eye(K)[yte]

for r in range(R):
    path = CACHE / f"rep_{r:02d}.npz"
    if os.path.exists(path):
        continue
    t0 = time.time()
    idx = RES[r]
    Xr, yr = Xtr[idx], ytr[idx]
    s = SEED + r + 1                                      # fresh model seed for each repetition

    tree = DecisionTreeClassifier(**tree_params, random_state=s).fit(Xr, yr)
    Pt = full_proba(tree.predict_proba(Xte), tree.classes_)
    t_tree = time.time() - t0

    t1 = time.time()
    boost = SAMME(T=bf["rounds"], depth=bf["depth"], K=K, seed=s).fit(Xr, yr)
    Ps = boost.predict_proba(Xte)
    t_boost = time.time() - t1

    t2 = time.time()
    bag = Bagging(B, tree_params, seed=s).fit(Xr, yr)
    Pb = bag.predict_proba(Xte)
    t_bag = time.time() - t2

    masked_acc = {
        "tree": float((tree.predict(Xtm) == yte).mean()),
        "bag": float((bag.predict(Xtm) == yte).mean()),
        "boost": float((boost.predict(Xtm) == yte).mean()),
    }
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, P_tree=Pt, P_bag=Pb, P_boost=Ps,
             masked_tree=masked_acc["tree"], masked_bag=masked_acc["bag"],
             masked_boost=masked_acc["boost"])
    os.replace(tmp, path)                                 # atomic: no half-written files
    print(f"rep {r+1:>2}/{R} done | tree {t_tree:.0f}s boost {t_boost:.0f}s bag {t_bag:.0f}s | "
          f"test acc tree {(Pt.argmax(1)==yte).mean():.3f} bag {(Pb.argmax(1)==yte).mean():.3f} "
          f"boost {(Ps.argmax(1)==yte).mean():.3f}", flush=True)

# ---- summary over all finished repetitions ----
files = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz") and ".tmp" not in f)
n_done = len(files)
if n_done < R:
    print(f"\nWARNING: only {n_done}/{R} repetitions finished. Re-run to continue.")
data = [np.load(CACHE / f) for f in files]

summary = {}
print(f"\n{'model':>9} {'bias^2':>8} {'variance':>9} {'MSE':>7} {'test acc':>16} {'masked acc':>11}")
for name, key in [("tree", "P_tree"), ("bagging", "P_bag"), ("boosting", "P_boost")]:
    P = np.stack([x[key] for x in data])                  # (R, N, K)
    pbar = P.mean(0)
    bias2 = float(((pbar - E) ** 2).sum(1).mean())
    var = float(((P - pbar[None]) ** 2).sum(2).mean())
    mse = float(((P - E[None]) ** 2).sum(2).mean())       # must equal bias^2 + variance
    accs = np.array([(p.argmax(1) == yte).mean() for p in P])
    mkey = {"tree": "masked_tree", "bagging": "masked_bag", "boosting": "masked_boost"}[name]
    macc = np.array([float(x[mkey]) for x in data])
    summary[name] = {"bias2": bias2, "variance": var, "mse": mse,
                     "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
                     "masked_acc_mean": float(macc.mean()), "masked_acc_std": float(macc.std(ddof=1))}
    print(f"{name:>9} {bias2:>8.4f} {var:>9.4f} {mse:>7.4f} "
          f"{accs.mean():>8.3f} +- {accs.std(ddof=1):.3f} {macc.mean():>8.3f}")
    print(f"{'':>9} check: bias2 + variance - MSE = {bias2 + var - mse:.2e}")

with (Q3_DIR / "q3a_results.json").open("w") as f:
    json.dump({"repetitions": n_done, "summary": summary}, f, indent=2)
print("\nSaved q3a_results.json")