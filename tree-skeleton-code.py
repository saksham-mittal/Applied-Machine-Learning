# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
import math
from operator import itemgetter

# Enter You Name Here
myname = "Saksham-Mittal" # or "Doe-Jane-"

def calc_entropy(pos, neg):
    e1 = 0
    if pos != 0:
        e1 = float(pos)/(pos + neg) * math.log(float(pos)/(pos + neg), 2)

    e2 = 0
    if neg != 0:
        e2 = float(neg)/(pos + neg) * math.log(float(neg)/(pos + neg), 2)

    return -(e1 + e2)

def find_entropy(attribute, result, mean_split):
    p1 = 0
    p2 = 0
    n1 = 0
    n2 = 0
    for i in range(len(attribute)):
        if attribute[i] <= mean_split:
            if result[i] == 1:
                p1 = p1 + 1
            else:
                n1 = n1 + 1
        else:
            if result[i] == 1:
                p2 = p2 + 1
            else:
                n2 = n2 + 1
    # print positive
    # print negative

    e1 = calc_entropy(p1, n1)
    e2 = calc_entropy(p2, n2)
    return (float(p1 + n1)/(p1 + n1 + p2 + n2) * e1) + (float(p2 + n2)/(p1 + n1 + p2 + n2) * e2)

def find_split(attribute, result, parent_entropy):
    sorted_attribute = attribute.copy()
    sorted_attribute.sort()

    max_inf_gain = 0
    mean_split_for_max_gain = 0

    for i in range(len(sorted_attribute) - 1):
        mean_split = (sorted_attribute[i] + sorted_attribute[i + 1])/float(2)
        entropy_from_current_split = find_entropy(attribute, result, mean_split)
        inf_gain = parent_entropy - entropy_from_current_split

        if inf_gain > max_inf_gain:
            max_inf_gain = inf_gain
            mean_split_for_max_gain = mean_split
    
    # Print maximum info gain for the current attribute
    print max_inf_gain
    return mean_split_for_max_gain, max_inf_gain

def run_split(data):
    labels = data[:, -1]
    p = 0
    n = 0
    columns = data.shape[1]

    for elem in labels:
        if elem == 1:
            p = p + 1
        else:
            n = n + 1

    mean_attr_list = []
    inf_gain_attr_list = []      
    for i in range(columns - 1):
        x, y = find_split(data[:, i], labels, calc_entropy(p, n))
        mean_attr_list.append(x)
        inf_gain_attr_list.append(y)
    
    max_inf_gain = max(inf_gain_attr_list)
    attr = inf_gain_attr_list.index(max_inf_gain)
    mean_split = mean_attr_list[attr]

    print "---------------------------"
    print mean_split
    print attr
    print "---------------------------"

# Implement your decision tree below
class DecisionTree():
    tree = {}

    def learn(self, training_set):
        # implement this function
        self.tree = {} 

    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        return result

def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print "Number of records: %d" % len(data)

    data = np.array(list(data)).astype("float")
    # print data

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    training_set = np.array(training_set)
    test_set = [x for i, x in enumerate(data) if i % K == 9]
    test_set = np.array(test_set)

    print training_set
    
    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn(training_set)

    run_split(training_set)

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print "accuracy: %.4f" % accuracy       
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
