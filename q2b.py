import json, time
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from samme import SAMME

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Q1"))
from common import load_data, get_seed

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xva, yva = d["X_val"], d["y_val"]
Xte, yte = d["X_test"], d["y_test"]
SEED = get_seed()
K = 40
depths = [1, 2, 3, 4]
rounds = [10, 25, 50, 100]


def prefix_acc(model, X, y, R):
    S = np.zeros((len(X), K))
    for h, a in list(zip(model.learners, model.alphas))[:R]:
        S[np.arange(len(X)), h.predict(X)] += a
    return float((S.argmax(1) == y).mean())


models, rows = {}, []
for dep in depths:
    t0 = time.time()
    m = SAMME(T=max(rounds), depth=dep, K=K, seed=SEED).fit(Xtr, ytr)
    models[dep] = m
    print(f"depth {dep}: fitted {len(m.learners)} learners in {time.time()-t0:.1f}s"
          f" | stopped: {getattr(m, 'stopped', None)}")
    for R in rounds:
        rows.append({"depth": dep, "rounds": R,
                     "learners_used": min(R, len(m.learners)),
                     "train": prefix_acc(m, Xtr, ytr, R),
                     "val": prefix_acc(m, Xva, yva, R),
                     "test": prefix_acc(m, Xte, yte, R)})

print(f"\n{'depth':>5} {'rounds':>6} {'used':>5} {'train':>7} {'val':>7} {'test':>7}")
for r in rows:
    print(f"{r['depth']:>5} {r['rounds']:>6} {r['learners_used']:>5} "
          f"{r['train']:>7.3f} {r['val']:>7.3f} {r['test']:>7.3f}")

# Selection rule (validation accuracy only): highest val; ties -> fewer rounds, then shallower depth.
best = max(rows, key=lambda r: (r["val"], -r["rounds"], -r["depth"]))
print("\nSELECTED:", {"depth": best["depth"], "rounds": best["rounds"]})

with open("boost_final.json", "w") as f:
    json.dump({"depth": best["depth"], "rounds": best["rounds"]}, f)
with open("q2b_results.json", "w") as f:
    json.dump({"rows": rows, "selected": {"depth": best["depth"], "rounds": best["rounds"]}}, f, indent=2)

# ---- diagnostics for the selected model ----
m = models[best["depth"]]
n_used = min(best["rounds"], len(m.alphas))
W = np.stack(m.w_hist[:n_used])                 # weights used to train each round, (n_used, 240)
err = np.array(m.err_hist[:n_used])
alpha = np.array(m.alphas[:n_used])
maxw = W.max(1)
top10 = np.sort(W, axis=1)[:, -10:].sum(1)
x = np.arange(1, n_used + 1)

fig, ax = plt.subplots(2, 2, figsize=(10, 6))
for a, y_, t in zip(ax.ravel(), [err, alpha, maxw, top10],
                    ["Weighted training error", "Learner weight (alpha)",
                     "Maximum sample weight", "Total weight of 10 heaviest images"]):
    a.plot(x, y_)
    a.set_title(t)
    a.set_xlabel("boosting round")
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("q2b_diagnostics.png", dpi=200)

print(f"\nRound 1: max weight {maxw[0]:.5f} (uniform = {1/len(ytr):.5f}), top-10 total {top10[0]:.4f}")
print(f"Round {n_used}: max weight {maxw[-1]:.5f}, top-10 total {top10[-1]:.4f}")
print(f"Weighted error: first {err[0]:.3f}, last {err[-1]:.3f}, max {err.max():.3f}")
print(f"Alpha: first {alpha[0]:.3f}, last {alpha[-1]:.3f}")
print("Saved boost_final.json, q2b_results.json, q2b_diagnostics.png")