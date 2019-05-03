import numpy as np
from sklearn.svm import SVC
import sys

training_set = np.genfromtxt("train_data.csv", dtype='float', delimiter=' ')
test_set = np.genfromtxt("test_data.csv", dtype='float', delimiter=' ')

training_set_reduced = np.empty(shape=[0, training_set.shape[1]])
for i in range(training_set.shape[0]):
    if training_set[i][0] == 1 or training_set[i][0] == 5:
        training_set_reduced = np.vstack((training_set_reduced, training_set[i, :]))

test_set_reduced = np.empty(shape=[0, test_set.shape[1]])
for i in range(test_set.shape[0]):
    if test_set[i][0] == 1 or test_set[i][0] == 5:
        test_set_reduced = np.vstack((test_set_reduced, test_set[i, :]))

# First column is the training labels
training_labels = training_set_reduced[:, 0]
# Removing labels from training_set_reduced
training_set_reduced = training_set_reduced[:, 1:]

test_labels = test_set_reduced[:, 0]
test_set_reduced = test_set_reduced[:, 1:]

# Q is the degree pf the polynomial
Q = int(sys.argv[1])
C = float(sys.argv[2])

clf = SVC(kernel='poly', degree=Q, C=C, coef0=1, gamma=1)
clf.fit(training_set_reduced, training_labels)

support_vecs = clf.n_support_
print "Number of support vectors used: %d" % (sum(support_vecs))

ans = clf.predict(test_set_reduced)

results = []
# Comparing ans with labels of test set
for c, instance in enumerate(test_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Test error: %.4f" % (1 - accuracy)

ans = clf.predict(training_set_reduced)
results = []
# Comparing ans with labels of train set
for c, instance in enumerate(training_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Train error: %.4f" % (1 - accuracy)