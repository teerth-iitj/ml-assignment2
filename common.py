import re
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROLL_NUMBER = "M26CSE010"   


def get_seed(roll=ROLL_NUMBER):
    digits = re.sub(r"\D", "", roll)      
    return int(digits)


def make_split(seed):
    """Per identity: 6 train, 2 validation, 2 test images."""
    rng = np.random.default_rng(seed)
    data = fetch_olivetti_faces(shuffle=False)
    X, y = data.data, data.target

    tr, va, te = [], [], []
    for k in range(40):
        idx = np.where(y == k)[0]
        idx = rng.permutation(idx)
        tr.extend(idx[:6])
        va.extend(idx[6:8])
        te.extend(idx[8:10])

    return np.array(tr), np.array(va), np.array(te)


def mask_images(X):
    """Zero the centred 16x16 block: rows and cols 24 to 39 (inclusive)."""
    Xm = X.copy().reshape(-1, 64, 64)
    Xm[:, 24:40, 24:40] = 0.0
    return Xm.reshape(-1, 4096)


def load_data():
    """Returns a dict with clean and masked data, using the saved split."""
    data = fetch_olivetti_faces(shuffle=False)
    X, y = data.data, data.target
    tr, va, te = make_split(get_seed())
    return {
        "X_train": X[tr], "y_train": y[tr],
        "X_val": X[va],   "y_val": y[va],
        "X_test": X[te],  "y_test": y[te],
        "X_test_masked": mask_images(X[te]),
        "idx_train": tr, "idx_val": va, "idx_test": te,
    }


if __name__ == "__main__":
    seed = get_seed()
    tr, va, te = make_split(seed)
    np.savez("split.npz", train=tr, val=va, test=te, seed=seed)
    print("Seed:", seed)
    print("Saved split.npz")

    d = load_data()
    print("Train:", d["X_train"].shape, " Val:", d["X_val"].shape,
          " Test:", d["X_test"].shape, " Masked test:", d["X_test_masked"].shape)
    print("Overlap train/val/test:",
          len(set(tr) & set(va)), len(set(tr) & set(te)), len(set(va) & set(te)))
    print("Masked block is zero:", bool((d["X_test_masked"].reshape(-1, 64, 64)[:, 24:40, 24:40] == 0).all()))
    print("Masked pixels differ from clean outside block:",
          bool(np.allclose(d["X_test_masked"].reshape(-1, 64, 64)[:, :24, :],
                           d["X_test"].reshape(-1, 64, 64)[:, :24, :])))
