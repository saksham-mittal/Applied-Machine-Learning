from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
# Create feature matrix
X = iris.data

# Create target vector
y = iris.target
label_ids = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

tsne_model = TSNE(n_components=2, random_state=535423)
X_normalize = tsne_model.fit_transform(X)

target_ids = [x for x in range(3)]

plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
colors = ['red', 'orange', 'purple']
for i, c, label in zip(target_ids, colors, label_ids):
    plt.scatter(X_normalize[y == i, 0], X_normalize[y == i, 1], c=c, label=label)

plt.legend()
plt.savefig("tsne.png", bbox_inches='tight')
plt.close()