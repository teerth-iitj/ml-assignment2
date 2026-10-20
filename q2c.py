import json, time
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

Q2_DIR = Path(__file__).resolve().parent
Q1_DIR = Q2_DIR.parent / "Q1"
sys.path.insert(0, str(Q1_DIR))

from samme import SAMME
from bagging import Bagging
from common import load_data, get_seed

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xva, yva = d["X_val"], d["y_val"]
Xte, yte = d["X_test"], d["y_test"]
SEED = get_seed()
K = 40

with (Q1_DIR / "frozen_tree.json").open() as f:
    tp = json.load(f)
tree_params = {"max_depth": tp["max_depth"], "min_samples_leaf": tp["min_samples_leaf"]}
with (Q2_DIR / "boost_final.json").open() as f:
    depth = json.load(f)["depth"]

# ---------- comparison table (each model fitted separately so timing is honest) ----------
rows, boost = [], {}
for T in [25, 50, 100]:
    t0 = time.time()
    mb = SAMME(T=T, depth=depth, K=K, seed=SEED).fit(Xtr, ytr)
    tb = time.time() - t0
    boost[T] = mb
    t0 = time.time()
    bg = Bagging(T, tree_params, seed=SEED).fit(Xtr, ytr)
    tg = time.time() - t0
    for name, m, tt in [(f"boosting (depth {depth})", mb, tb), ("bagging (frozen tree)", bg, tg)]:
        rows.append({"model": name, "learners": T, "train": m.score(Xtr, ytr),
                     "val": m.score(Xva, yva), "test": m.score(Xte, yte), "time_s": tt})

print(f"{'model':>24} {'T':>4} {'train':>7} {'val':>7} {'test':>7} {'time(s)':>8}")
for r in rows:
    print(f"{r['model']:>24} {r['learners']:>4} {r['train']:>7.3f} {r['val']:>7.3f} "
          f"{r['test']:>7.3f} {r['time_s']:>8.1f}")

# ---------- inspect a heavily weighted training face (100-round model) ----------
m = boost[100]
W = np.stack(m.w_hist)                    # (rounds, 240): weights each round was trained on
H = np.stack(m.pred_hist)                 # (rounds, 240): each learner's prediction on train faces
miss = H != ytr[None, :]
peak = W.max(0)
i = int(peak.argmax())
peak_round = int(W[:, i].argmax()) + 1

print(f"\nTop-5 peak weights (train idx: peak): "
      + ", ".join(f"{j}: {peak[j]:.4f}" for j in np.argsort(-peak)[:5]))
print(f"Chosen face: train idx {i}, true id {ytr[i]}, peak weight {peak[i]:.4f} "
      f"at round {peak_round} (uniform = {1/len(ytr):.5f}, ratio {peak[i]*len(ytr):.1f}x)")
print("Weight at rounds 1,10,25,50,75,100:",
      [round(float(W[r-1, i]), 5) for r in [1, 10, 25, 50, 75, 100] if r <= len(W)])
print(f"Misclassified in {int(miss[:, i].sum())} of {len(W)} rounds "
      f"(average over all faces: {miss.mean()*len(W):.1f})")
wrong = Counter(H[miss[:, i], i].tolist()).most_common(3)
print("Most common wrong labels (id, count):", wrong)
print("Final ensemble prediction:", int(m.predict(Xtr[i:i+1])[0]), "| true:", int(ytr[i]))

ratio_miss = [W[t+1, i] / W[t, i] for t in range(len(W)-1) if miss[t, i]]
ratio_hit = [W[t+1, i] / W[t, i] for t in range(len(W)-1) if not miss[t, i]]
print(f"Next-round weight ratio after a MISS: mean {np.mean(ratio_miss):.2f} | "
      f"after a HIT: mean {np.mean(ratio_hit):.2f}")

with (Q2_DIR / "q2c_results.json").open("w") as f:
    json.dump({"rows": rows, "face": {"train_idx": i, "true": int(ytr[i]),
               "peak_weight": float(peak[i]), "peak_round": peak_round,
               "n_missed": int(miss[:, i].sum()),
               "wrong_labels": [(int(a), int(b)) for a, b in wrong]}}, f, indent=2)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
r = np.arange(1, len(W) + 1)
ax[0].plot(r, W[:, i], lw=1)
ax[0].scatter(r[miss[:, i]], W[miss[:, i], i], s=10, c="red", label="misclassified this round")
ax[0].axhline(1/len(ytr), color="gray", ls="--", lw=0.8, label="uniform weight")
ax[0].set_xlabel("boosting round"); ax[0].set_ylabel("sample weight")
ax[0].set_title(f"Weight of train face {i}"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
ax[1].imshow(Xtr[i].reshape(64, 64), cmap="gray"); ax[1].axis("off")
ax[1].set_title(f"Train face {i} (true id {ytr[i]})", fontsize=9)
wl = wrong[0][0] if wrong else ytr[i]
ax[2].imshow(Xtr[ytr == wl][0].reshape(64, 64), cmap="gray"); ax[2].axis("off")
ax[2].set_title(f"Most common wrong label (id {wl})", fontsize=9)
plt.tight_layout(); plt.savefig(Q2_DIR / "q2c_heavy_face.png", dpi=200)
print("Saved q2c_results.json, q2c_heavy_face.png")