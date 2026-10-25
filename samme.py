import numpy as np
from sklearn.tree import DecisionTreeClassifier


def samme_round(h_pred, y, w, K):
    """One SAMME weight update. Returns (err, alpha, new_w, status).
    status: 'ok', 'perfect' or 'not_better_than_random'."""
    miss = (h_pred != y)
    err = float(np.sum(w[miss]) / np.sum(w))
    if err >= 1 - 1.0 / K:
        return err, 0.0, w.copy(), "not_better_than_random"
    if err <= 1e-12:
        return err, 0.0, w.copy(), "perfect"
    alpha = np.log((1 - err) / err) + np.log(K - 1)
    new_w = w * np.exp(alpha * miss)
    new_w = new_w / new_w.sum()
    return err, float(alpha), new_w, "ok"


class SAMME:
    def __init__(self, T, depth, K=40, seed=0):
        self.T, self.depth, self.K, self.seed = T, depth, K, seed

    def fit(self, X, y):
        n = len(X)
        w = np.full(n, 1.0 / n)
        self.learners, self.alphas = [], []
        self.err_hist, self.w_hist, self.pred_hist = [], [], []
        rng = np.random.default_rng(self.seed)
        for t in range(self.T):
            h = DecisionTreeClassifier(max_depth=self.depth,
                                       random_state=int(rng.integers(0, 2**31 - 1)))
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            err, alpha, w_new, status = samme_round(pred, y, w, self.K)
            self.pred_hist.append(pred)
            self.w_hist.append(w.copy())
            self.err_hist.append(err)
            if status != "ok":
                self.stopped = (t, status)
                if status == "perfect":          
                    self.learners.append(h)
                    self.alphas.append(np.log((1 - 1e-12) / 1e-12) + np.log(self.K - 1))
                break                            
            self.learners.append(h)
            self.alphas.append(alpha)
            w = w_new
        self.final_w = w
        return self

    def scores(self, X):
        S = np.zeros((X.shape[0], self.K))
        for h, a in zip(self.learners, self.alphas):
            S[np.arange(X.shape[0]), h.predict(X)] += a
        return S

    def predict(self, X):
        return self.scores(X).argmax(1)

    def predict_proba(self, X):
        S = self.scores(X)
        return S / max(sum(self.alphas), 1e-12)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


if __name__ == "__main__":
    # Toy problem: 3 classes, 10 samples, one feature. A stump cannot get everything right.
    K = 3
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 0])      # last point (x=10) is a class-0 outlier
    n = len(y)
    w = np.full(n, 1.0 / n)

    h = DecisionTreeClassifier(max_depth=1, random_state=0)
    h.fit(X, y, sample_weight=w)
    pred = h.predict(X)
    err, alpha, w_new, status = samme_round(pred, y, w, K)

    np.set_printoptions(precision=4, suppress=True)
    print("K =", K, " n =", n)
    print("true labels      :", y)
    print("initial weights  :", w)
    print("weak learner pred:", pred)
    print("misclassified    :", (pred != y).astype(int))
    print(f"weighted error   : {err:.4f}   (random-guess threshold 1-1/K = {1-1/K:.4f})")
    print(f"alpha            : log((1-err)/err) + log(K-1) = {alpha:.4f}")
    print("updated weights  :", w_new, " sum =", round(float(w_new.sum()), 6))
    print("status           :", status)

    # Hand check of the update rule
    chk = w * np.exp(alpha * (pred != y)); chk /= chk.sum()
    print("hand-check matches:", bool(np.allclose(chk, w_new)))
