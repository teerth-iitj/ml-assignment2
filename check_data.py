from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces(shuffle=False)
X, y = data.data, data.target

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Pixel range:", X.min(), "to", X.max())
print("Number of identities:", len(set(y)))
print("Images per identity:", set((y == k).sum() for k in range(40)))