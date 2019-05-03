import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = np.genfromtxt('spam.csv', delimiter=' ')

# Split training/test sets
indices = np.random.permutation(data.shape[0])
split_val = int(0.7 * data.shape[0])
training_idx, test_idx = indices[:split_val], indices[split_val:]
training_set, test_set = data[training_idx,:], data[test_idx,:]

# Extracting labels from training and test set
training_labels = training_set[:, -1]
test_labels = test_set[:, -1]

# Dropping the last column from training and test set
training_set = training_set[:, :-1]
test_set = test_set[:, :-1]
print "Data Preprocessing completed"

# Using sklearn random forest classifier
clf = RandomForestClassifier(n_estimators=100)
print "Data classification completed"

# Fitting the training data
clf.fit(training_set, training_labels)
print "Data fitting completed"

ans = clf.predict(test_set)
print "Data prediction completed"

results = []
for c, instance in enumerate(test_labels):
    results.append(instance == ans[c])

# Accuracy
accuracy = float(results.count(True))/float(len(results))
print "accuracy: %.4f" % (accuracy)