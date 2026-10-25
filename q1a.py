import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data, get_seed

d = load_data()
Xtr, ytr = d["X_train"], d["y_train"]
Xva, yva = d["X_val"], d["y_val"]
Xte, yte = d["X_test"], d["y_test"]

SEED = get_seed()
depths = [2, 4, 8, 16, None]
leaves = [1, 2, 4]

def leaf_stats(tree, Xtr):
    t = tree.tree_
    is_leaf = t.children_left == -1
    n_leaves = int(is_leaf.sum())
    # training samples per leaf
    leaf_ids = tree.apply(Xtr)
    counts = np.array([(leaf_ids == l).sum() for l in np.where(is_leaf)[0]])
    return {
        "n_leaves": n_leaves,
        "mean_samples_per_leaf": float(counts.mean()),
        "frac_singleton_leaves": float((counts == 1).mean()),
        "actual_depth": int(tree.get_depth()),
    }

rows = []
for md in depths:
    for ml in leaves:
        clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=ml,
                                     random_state=SEED)
        clf.fit(Xtr, ytr)
        row = {
            "max_depth": md, "min_samples_leaf": ml,
            "train_acc": clf.score(Xtr, ytr),
            "val_acc": clf.score(Xva, yva),
            "test_acc": clf.score(Xte, yte),
        }
        row.update(leaf_stats(clf, Xtr))
        rows.append(row)

print(f"{'depth':>6} {'leaf':>5} {'train':>7} {'val':>7} {'test':>7} "
      f"{'#leaves':>8} {'mean/leaf':>10} {'frac1':>7} {'real_d':>7}")
for r in rows:
    print(f"{str(r['max_depth']):>6} {r['min_samples_leaf']:>5} "
          f"{r['train_acc']:>7.3f} {r['val_acc']:>7.3f} {r['test_acc']:>7.3f} "
          f"{r['n_leaves']:>8} {r['mean_samples_per_leaf']:>10.2f} "
          f"{r['frac_singleton_leaves']:>7.3f} {r['actual_depth']:>7}")

# Selection rule: validation accuracy ONLY.
# Ties are broken by the first row in grid order
best = max(rows, key=lambda r: r["val_acc"])
print("\nFROZEN SETTING:", {"max_depth": best["max_depth"],
                            "min_samples_leaf": best["min_samples_leaf"]})
print(best)

with open("frozen_tree.json", "w") as f:
    json.dump({"max_depth": best["max_depth"],
               "min_samples_leaf": best["min_samples_leaf"],
               "random_state": SEED}, f, indent=2)
with open("q1a_results.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nSaved frozen_tree.json and q1a_results.json")
