import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bagging import Bagging
from common import load_data, get_seed

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xva, yva = d["X_val"], d["y_val"]
SEED = get_seed()
K = 40
Bs = [1, 5, 10, 25, 50, 100, 200]

with open("frozen_tree.json") as f:
    tp = json.load(f)
tree_params = {"max_depth": tp["max_depth"], "min_samples_leaf": tp["min_samples_leaf"]}

m = Bagging(200, tree_params, seed=SEED).fit(Xtr, ytr)
P_tr = m.tree_preds(Xtr)      # (200, 240)
P_va = m.tree_preds(Xva)      # (200, 80)
n = len(Xtr)

def votes(P):
    V = np.zeros((P.shape[1], K))
    for b in range(P.shape[0]):
        V[np.arange(P.shape[1]), P[b]] += 1
    return V

rows = []
for B in Bs:
    # --- validation accuracy (used for selecting B) ---
    Vva = votes(P_va[:B])
    val_acc = float((Vva.argmax(1) == yva).mean())

    # --- out-of-bag accuracy on the training set ---
    oob = ~np.stack(m.in_bag[:B])              # (B, n): True where tree did NOT see sample
    Voob = np.zeros((n, K))
    for b in range(B):
        idx = np.where(oob[b])[0]
        Voob[idx, P_tr[b][idx]] += 1
    covered = Voob.sum(1) > 0
    oob_acc = float((Voob.argmax(1) == ytr)[covered].mean()) if covered.any() else float("nan")
    oob_cov = float(covered.mean())

    # --- average pairwise disagreement between trees (on validation set) ---
    if B > 1:
        agree_pairs = (Vva * (Vva - 1) / 2).sum(1)      # pairs that voted the same class
        disagree = float((1 - agree_pairs / (B * (B - 1) / 2)).mean())
    else:
        disagree = float("nan")

    # --- mean vote margin: winning fraction minus runner-up fraction (validation) ---
    F = np.sort(Vva / B, axis=1)
    margin = float((F[:, -1] - F[:, -2]).mean())

    rows.append({"B": B, "val_acc": val_acc, "oob_acc": oob_acc,
                 "oob_coverage": oob_cov, "disagreement": disagree,
                 "vote_margin": margin})

print(f"{'B':>4} {'val':>7} {'OOB':>7} {'OOBcov':>7} {'disagree':>9} {'margin':>7}")
for r in rows:
    print(f"{r['B']:>4} {r['val_acc']:>7.3f} {r['oob_acc']:>7.3f} {r['oob_coverage']:>7.3f} "
          f"{r['disagreement']:>9.3f} {r['vote_margin']:>7.3f}")

# Selection rule: highest validation accuracy; ties go to the smaller B.
best = max(rows, key=lambda r: (r["val_acc"], -r["B"]))
print("\nSELECTED B:", best["B"])

with open("q1c_results.json", "w") as f:
    json.dump({"rows": rows, "selected_B": best["B"]}, f, indent=2)
with open("final_B.json", "w") as f:
    json.dump({"B": best["B"]}, f)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, key, title in zip(ax, ["oob_acc", "disagreement", "vote_margin"],
                         ["Out-of-bag accuracy", "Mean pairwise disagreement", "Mean vote margin"]):
    a.plot([r["B"] for r in rows], [r[key] for r in rows], "o-")
    a.set_xscale("log"); a.set_xlabel("B (number of trees)"); a.set_title(title)
    a.axvline(best["B"], color="gray", ls="--", lw=0.8)
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("q1c_diagnostics.png", dpi=200)
print("Saved q1c_results.json, final_B.json, q1c_diagnostics.png")