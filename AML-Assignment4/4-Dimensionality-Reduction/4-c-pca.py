from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.samples_generator.make_swiss_roll(random_state=0, n_samples=1500)

pca_model = PCA(n_components=2)
X = pca_model.fit(X).transform(X)

plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.savefig("4-c-pca.png", bbox_inches='tight')
plt.close()