import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Bagging:
    def __init__(self, B, tree_params, n_classes=40, seed=0):
        self.B = B
        self.tree_params = tree_params
        self.K = n_classes
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.trees, self.in_bag, self.n_distinct = [], [], []
        for b in range(self.B):
            idx = rng.integers(0, n, size=n)          # bootstrap, with replacement
            tree = DecisionTreeClassifier(
                **self.tree_params,
                random_state=int(rng.integers(0, 2**31 - 1)))
            tree.fit(X[idx], y[idx])
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            self.trees.append(tree)
            self.in_bag.append(mask)
            self.n_distinct.append(int(mask.sum()))
        return self

    def tree_preds(self, X):
        # (B, N) matrix of each tree's predicted label
        return np.stack([t.predict(X) for t in self.trees])

    def vote_counts(self, X):
        P = self.tree_preds(X)                         # (B, N)
        V = np.zeros((X.shape[0], self.K))
        for b in range(P.shape[0]):
            V[np.arange(X.shape[0]), P[b]] += 1
        return V                                       # (N, K) vote counts

    def predict(self, X):
        return self.vote_counts(X).argmax(axis=1)

    def predict_proba(self, X):
        return self.vote_counts(X) / self.B

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


if __name__ == "__main__":
    from common import load_data, get_seed

    d = load_data()
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]
    SEED = get_seed()

    with open("frozen_tree.json") as f:
        tp = json.load(f)
    tree_params = {"max_depth": tp["max_depth"],
                   "min_samples_leaf": tp["min_samples_leaf"]}

    n = len(Xtr)
    expected = n * (1 - (1 - 1 / n) ** n)
    print(f"Theoretical expected distinct per bootstrap: {expected:.2f} (n={n})\n")

    print(f"{'B':>4} {'train':>7} {'val':>7} {'test':>7} {'avg_distinct':>13}")
    rows = []
    for B in [1, 5, 10, 25, 50, 100, 200]:
        m = Bagging(B, tree_params, seed=SEED).fit(Xtr, ytr)
        r = {"B": B, "train": m.score(Xtr, ytr), "val": m.score(Xva, yva),
             "test": m.score(Xte, yte),
             "avg_distinct": float(np.mean(m.n_distinct))}
        rows.append(r)
        print(f"{B:>4} {r['train']:>7.3f} {r['val']:>7.3f} {r['test']:>7.3f} "
              f"{r['avg_distinct']:>13.2f}")

    with open("q1b_results.json", "w") as f:
        json.dump({"expected_distinct": expected, "rows": rows}, f, indent=2)
    print("\nSaved q1b_results.json")