from sklearn.neural_network import MLPClassifier

# 3 input fitur (misal tabel AND untuk 3 variabel masukan)
X = [
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]
y = [0, 0, 0, 0, 0, 0, 0, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

print(clf.predict([[1., 1., 1.], [0., 1., 0.]]))
print([coef.shape for coef in clf.coefs_])