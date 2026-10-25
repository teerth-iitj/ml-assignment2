import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from bagging import Bagging
from common import load_data, get_seed

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xte, yte = d["X_test"], d["y_test"]
Xtm = d["X_test_masked"]
SEED = get_seed()

with open("frozen_tree.json") as f:
    tp_cfg = json.load(f)
with open("final_B.json") as f:
    B = json.load(f)["B"]
tree_params = {"max_depth": tp_cfg["max_depth"],
               "min_samples_leaf": tp_cfg["min_samples_leaf"]}

tree = DecisionTreeClassifier(**tree_params, random_state=SEED).fit(Xtr, ytr)
bag = Bagging(B, tree_params, seed=SEED).fit(Xtr, ytr)


def evaluate(X, y):
    tpred = tree.predict(X)
    tconf = tree.predict_proba(X).max(1)           # winning class probability
    V = bag.vote_counts(X)
    bpred = V.argmax(1)
    bconf = V.max(1) / B                           # winning vote fraction
    return tpred, tconf, V, bpred, bconf


def summarise(name, pred, conf, y):
    ok = pred == y
    return {"model": name, "acc": float(ok.mean()), "mean_conf": float(conf.mean()),
            "conf_correct": float(conf[ok].mean()) if ok.any() else float("nan"),
            "conf_wrong": float(conf[~ok].mean()) if (~ok).any() else float("nan")}


results = {}
for setname, X in [("clean", Xte), ("masked", Xtm)]:
    tpred, tconf, V, bpred, bconf = evaluate(X, yte)
    results[setname] = [summarise("tree", tpred, tconf, yte),
                        summarise(f"bagging B={B}", bpred, bconf, yte)]

print(f"{'set':>7} {'model':>14} {'acc':>7} {'conf':>7} {'conf|ok':>8} {'conf|wrong':>11}")
for setname, rs in results.items():
    for r in rs:
        print(f"{setname:>7} {r['model']:>14} {r['acc']:>7.3f} {r['mean_conf']:>7.3f} "
              f"{r['conf_correct']:>8.3f} {r['conf_wrong']:>11.3f}")

# ---- example faces on the CLEAN test set ----
tpred, tconf, V, bpred, bconf = evaluate(Xte, yte)
fixed = np.where((tpred != yte) & (bpred == yte))[0]
both_wrong = np.where((tpred != yte) & (bpred != yte))[0]
print(f"\nBagging corrects tree: {len(fixed)} faces | both wrong: {len(both_wrong)} faces")

# Selection rules (fixed in advance):
fixed_sel = fixed[np.argsort(-V[fixed, yte[fixed]])][:2]
wrong_sel = both_wrong[np.argsort(-bconf[both_wrong])][:2]


def describe(kind, i):
    top = np.argsort(-V[i])[:3]
    top_str = ", ".join(f"id {c}: {int(V[i, c])}" for c in top)
    print(f"[{kind}] test idx {i} | true id {yte[i]} | tree pred {tpred[i]} | "
          f"bag pred {bpred[i]} | votes for true id: {int(V[i, yte[i]])}/{B} | top votes: {top_str}")
    return {"kind": kind, "test_idx": int(i), "true": int(yte[i]), "tree_pred": int(tpred[i]),
            "bag_pred": int(bpred[i]), "votes_true": int(V[i, yte[i]]),
            "top_votes": {int(c): int(V[i, c]) for c in top}}


print()
cases = [describe("corrected", i) for i in fixed_sel] + [describe("both wrong", i) for i in wrong_sel]

with open("q1d_results.json", "w") as f:
    json.dump({"B": B, "results": results, "cases": cases}, f, indent=2)


fig, ax = plt.subplots(len(cases), 4, figsize=(8, 2.1 * len(cases)))
for r, c in enumerate(cases):
    i = c["test_idx"]
    imgs = [Xte[i], Xtr[ytr == c["true"]][0], Xtr[ytr == c["tree_pred"]][0], Xtr[ytr == c["bag_pred"]][0]]
    titles = [f"{c['kind']}: test", f"true (id {c['true']})",
              f"tree pred (id {c['tree_pred']})", f"bag pred (id {c['bag_pred']})"]
    for k in range(4):
        ax[r, k].imshow(imgs[k].reshape(64, 64), cmap="gray")
        ax[r, k].set_title(titles[k], fontsize=7)
        ax[r, k].axis("off")
plt.tight_layout()
plt.savefig("q1d_cases.png", dpi=200)
print("\nSaved q1d_results.json, q1d_cases.png")
