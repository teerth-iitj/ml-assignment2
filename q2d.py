import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

Q2_DIR = Path(__file__).resolve().parent
Q1_DIR = Q2_DIR.parent / "Q1"
sys.path.insert(0, str(Q1_DIR))

from samme import SAMME
from bagging import Bagging
from common import load_data, get_seed

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xte, yte = d["X_test"], d["y_test"]
Xtm = d["X_test_masked"]
SEED = get_seed()
K = 40

with (Q1_DIR / "frozen_tree.json").open() as f:
    tp = json.load(f)
with (Q1_DIR / "final_B.json").open() as f:
    B = json.load(f)["B"]
with (Q2_DIR / "boost_final.json").open() as f:
    bf = json.load(f)
tree_params = {"max_depth": tp["max_depth"], "min_samples_leaf": tp["min_samples_leaf"]}

tree = DecisionTreeClassifier(**tree_params, random_state=SEED).fit(Xtr, ytr)
bag = Bagging(B, tree_params, seed=SEED).fit(Xtr, ytr)
boost = SAMME(T=bf["rounds"], depth=bf["depth"], K=K, seed=SEED).fit(Xtr, ytr)


def outputs(X):
    tp_ = tree.predict_proba(X)
    Vb = bag.vote_counts(X)
    qb = Vb / B
    qs = boost.predict_proba(X)
    return {"tree": (tp_.argmax(1), tp_.max(1), tp_),
            "bagging": (qb.argmax(1), qb.max(1), qb),
            "boosting": (qs.argmax(1), qs.max(1), qs)}, Vb


res = {}
out_clean, Vb_clean = outputs(Xte)
out_mask, Vb_mask = outputs(Xtm)

print(f"{'model':>9} | {'clean acc':>9} {'clean conf':>10} | {'mask acc':>8} {'mask conf':>9}")
table = []
for name in ["tree", "bagging", "boosting"]:
    pc, cc, _ = out_clean[name]
    pm, cm, _ = out_mask[name]
    r = {"model": name, "clean_acc": float((pc == yte).mean()), "clean_conf": float(cc.mean()),
         "masked_acc": float((pm == yte).mean()), "masked_conf": float(cm.mean())}
    table.append(r)
    print(f"{name:>9} | {r['clean_acc']:>9.3f} {r['clean_conf']:>10.3f} | "
          f"{r['masked_acc']:>8.3f} {r['masked_conf']:>9.3f}")

# ---- find faces where bagging and boosting respond differently to the mask ----
bc, bm = out_clean["bagging"][0], out_mask["bagging"][0]
sc, sm = out_clean["boosting"][0], out_mask["boosting"][0]
bag_ok_c, bag_ok_m = bc == yte, bm == yte
bst_ok_c, bst_ok_m = sc == yte, sm == yte

# bagging stays correct on the masked image, boosting flips from correct to wrong
cand_A = np.where(bag_ok_c & bag_ok_m & bst_ok_c & ~bst_ok_m)[0]
# boosting stays correct on the masked image, bagging flips from correct to wrong
cand_B = np.where(bst_ok_c & bst_ok_m & bag_ok_c & ~bag_ok_m)[0]
print(f"\nBagging robust / boosting breaks: {len(cand_A)} faces | "
      f"Boosting robust / bagging breaks: {len(cand_B)} faces")

# Selection rule (fixed in advance): prefer group A, then B; within the group choose the face
# with the largest drop in the failing model's true-class score.
if len(cand_A):
    group, kind, failing = cand_A, "bagging robust, boosting breaks", "boosting"
elif len(cand_B):
    group, kind, failing = cand_B, "boosting robust, bagging breaks", "bagging"
else:
    group, kind, failing = np.array([], dtype=int), "none found", None

if failing is not None:
    q_c = out_clean[failing][2][group, yte[group]]
    q_m = out_mask[failing][2][group, yte[group]]
    i = int(group[np.argmax(q_c - q_m)])
    print(f"\nChosen face: test idx {i}, true id {yte[i]} ({kind})")
    for name in ["tree", "bagging", "boosting"]:
        pc, cc, qc = out_clean[name]
        pm, cm, qm = out_mask[name]
        print(f"  {name:>8}: clean pred {pc[i]} (conf {cc[i]:.3f}, true-class {qc[i, yte[i]]:.3f}) -> "
              f"masked pred {pm[i]} (conf {cm[i]:.3f}, true-class {qm[i, yte[i]]:.3f})")

    # bagging: how many trees changed their vote because of the mask
    Pc, Pm = bag.tree_preds(Xte[i:i+1])[:, 0], bag.tree_preds(Xtm[i:i+1])[:, 0]
    print(f"  bagging: trees changing vote under mask: {int((Pc != Pm).sum())}/{B}; "
          f"votes for true id clean {int((Pc == yte[i]).sum())} -> masked {int((Pm == yte[i]).sum())}")
    # boosting: the heavy-weight learners' behaviour on this face
    S_c = boost.scores(Xte[i:i+1])[0]; S_m = boost.scores(Xtm[i:i+1])[0]
    top_c, top_m = np.argsort(-S_c)[:3], np.argsort(-S_m)[:3]
    print("  boosting top-3 clean :", [(int(c), round(float(S_c[c]), 2)) for c in top_c])
    print("  boosting top-3 masked:", [(int(c), round(float(S_m[c]), 2)) for c in top_m])
    print(f"  boosting total alpha = {sum(boost.alphas):.2f}")

    with (Q2_DIR / "q2d_results.json").open("w") as f:
        json.dump({"table": table, "face": {"test_idx": i, "true": int(yte[i]), "kind": kind}}, f, indent=2)

    fig, ax = plt.subplots(1, 4, figsize=(10, 2.8))
    ims = [Xte[i], Xtm[i], Xtr[ytr == yte[i]][0], Xtr[ytr == out_mask[failing][0][i]][0]]
    ts = ["clean test", "masked test", f"true id {yte[i]}", f"{failing} masked pred {out_mask[failing][0][i]}"]
    for a, im, t in zip(ax, ims, ts):
        a.imshow(im.reshape(64, 64), cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
    plt.tight_layout(); plt.savefig(Q2_DIR / "q2d_face.png", dpi=200)
else:
    with (Q2_DIR / "q2d_results.json").open("w") as f:
        json.dump({"table": table, "face": None}, f, indent=2)

print("\nSaved q2d_results.json" + (", q2d_face.png" if failing is not None else ""))