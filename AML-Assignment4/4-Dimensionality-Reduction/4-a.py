from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
# Create feature matrix
X = iris.data
# Create target vector
y = iris.target

pca_model = PCA(n_components=2)
X = pca_model.fit(X).transform(X)

plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
fmtr = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.colorbar(ticks=[0, 1, 2], format=fmtr)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.savefig("iris.png", bbox_inches='tight')
plt.close()