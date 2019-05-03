from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.samples_generator.make_swiss_roll(random_state=0, n_samples=1500)

tsne_model = TSNE(n_components=2, random_state=535423)

X_normalize = tsne_model.fit_transform(X)

plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
plt.scatter(X_normalize[:, 0], X_normalize[:, 1], c=y)

plt.savefig("4-c-tsne.png", bbox_inches='tight')
plt.close()