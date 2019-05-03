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

num_points = [50, 100, 200, 800, training_set_reduced.shape[0]]
for n in num_points:
    # Taking first n points from dataset
    training_reduced = training_set_reduced[:n]
    training_reduced_labels = training_labels[:n]

    clf = SVC(kernel='linear')
    clf.fit(training_reduced, training_reduced_labels)

    # Number of support vectors used
    support_vecs = clf.n_support_
    print "number of support vectors used for %d points: %d" % (n, sum(support_vecs))
    ans = clf.predict(test_set_reduced)

    results = []
    # Comparing ans with labels of test set
    for c, instance in enumerate(test_labels):
        results.append(instance == ans[c])

    # Finding accuracy
    accuracy = float(results.count(True))/float(len(results))
    print "accuracy over %d points: %.4f" % (n, accuracy)
    