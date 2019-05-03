import numpy as np
from sklearn.svm import SVC
import sys

training_set = np.genfromtxt("gisette_train.csv", dtype='int', delimiter=' ')
training_labels = np.genfromtxt("gisette_train_labels.csv", dtype = 'int', delimiter = ' ')
test_set = np.genfromtxt("gisette_test.csv", dtype='int', delimiter=' ')
test_set_labels = np.genfromtxt("gisette_test_labels.csv", dtype = 'int', delimiter = ' ')
print "Data loaded"

""" Using linear kernel """
clf = SVC(kernel='linear')
clf.fit(training_set, training_labels)
support_vecs = clf.n_support_
print "Number of support vectors used in linear kernel: %d" % (sum(support_vecs))

ans = clf.predict(training_set)
results = []
# Comparing ans with labels of train set
for c, instance in enumerate(training_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Train error using linear kernel: %.4f" % (1 - accuracy)

ans = clf.predict(test_set)
results = []
# Comparing ans with labels of test set
for c, instance in enumerate(test_set_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Test error using linear kernel: %.4f" % (1 - accuracy)
""" ---------------- """

""" Using RBF kernel """
clf = SVC(kernel='rbf', gamma=0.001)
clf.fit(training_set, training_labels)
support_vecs = clf.n_support_
print "Number of support vectors used in rbf kernel: %d" % (sum(support_vecs))

ans = clf.predict(training_set)
results = []
# Comparing ans with labels of train set
for c, instance in enumerate(training_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Train error using rbf kernel: %.4f" % (1 - accuracy)

ans = clf.predict(test_set)
results = []
# Comparing ans with labels of test set
for c, instance in enumerate(test_set_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Test error using rbf kernel: %.4f" % (1 - accuracy)
""" ----------------------- """

""" Using polynomial kernel """
clf = SVC(kernel='poly', degree=2, coef0=1, gamma=1)
clf.fit(training_set, training_labels)
support_vecs = clf.n_support_
print "Number of support vectors used in polynomial kernel: %d" % (sum(support_vecs))

ans = clf.predict(training_set)
results = []
# Comparing ans with labels of train set
for c, instance in enumerate(training_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Train error using polynomial kernel: %.4f" % (1 - accuracy)

ans = clf.predict(test_set)
results = []
# Comparing ans with labels of test set
for c, instance in enumerate(test_set_labels):
    results.append(instance == ans[c])

# Finding accuracy
accuracy = float(results.count(True))/float(len(results))
print "Test error using polynomial kernel: %.4f" % (1 - accuracy)
""" ------------------ """
