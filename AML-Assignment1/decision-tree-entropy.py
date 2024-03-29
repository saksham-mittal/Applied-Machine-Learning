# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
import math
from operator import itemgetter

# Enter You Name Here
myname = "Saksham-Mittal" # or "Doe-Jane-"

# Implement your decision tree below
class Tree():
    def __init__(self): 
        self.leftNode = None
        self.rightNode = None
        self.attrValue = None
        self.mean_split = None
        self.isLeaf = False
        self.predictedVal = None

def calc_entropy(pos, neg):
    e1 = 0
    if pos != 0:
        e1 = (float(pos)/(pos + neg)) * math.log(float(pos)/(pos + neg), 2)

    e2 = 0
    if neg != 0:
        e2 = (float(neg)/(pos + neg)) * math.log(float(neg)/(pos + neg), 2)

    return -(e1 + e2)

def find_split(attribute, label, parent_entropy):
    max_inf_gain = 0
    mean_split_for_max_gain = 0

    rows = attribute.shape[0]

    min_attr = min(attribute)
    max_attr = max(attribute)

    inc = (max_attr - min_attr)/float(15)
    for i in range(15):
        mean_split = min_attr + inc * i
        p1 = 0
        p2 = 0
        n1 = 0
        n2 = 0

        for j in range(rows):
            if attribute[j] <= mean_split:
                if label[j] == 1:
                    p1 = p1 + 1
                else:
                    n1 = n1 + 1
            else:
                if label[j] == 1:
                    p2 = p2 + 1
                else:
                    n2 = n2 + 1

        entropy_from_current_split = 0
        e1 = calc_entropy(p1, n1)
        e2 = calc_entropy(p2, n2)

        if (p1 + n1 + p2 + n2) != 0:
            entropy_from_current_split = (float(p1 + n1)/(p1 + n1 + p2 + n2)) * e1 + (float(p2 + n2)/(p1 + n1 + p2 + n2)) * e2

        if max_inf_gain < (parent_entropy - entropy_from_current_split):
            max_inf_gain = parent_entropy - entropy_from_current_split
            mean_split_for_max_gain = mean_split
    
    # Return maximum info gain for the current attribute and split
    return mean_split_for_max_gain, max_inf_gain

def run_split(node, data):
    p = 0
    n = 0
    columns = data.shape[1]
    rows = data.shape[0]

    for elem in data[:, -1]:
        if elem == 1:
            p = p + 1
        else:
            n = n + 1

    e = calc_entropy(p, n)
    if e == 0 or columns == 1:
        node.isLeaf = True
        if p == max(p, n):
            node.predictedVal = 1
        else:
            node.predictedVal = 0
        return

    mean_attr_list = []
    inf_gain_attr_list = []

    for i in range(columns - 1):
        x, y = find_split(data[:, i], data[:, -1], e)
        mean_attr_list.append(x)
        inf_gain_attr_list.append(y)
    
    max_inf_gain = max(inf_gain_attr_list)
    attr = inf_gain_attr_list.index(max_inf_gain)
    mean_split = mean_attr_list[attr]

    node.attrValue = attr
    node.mean_split = mean_split
    node.leftNode = Tree()
    node.rightNode = Tree()

    # print "---------------------------"
    # print e
    # print max_inf_gain
    # print mean_split
    # print attr
    # print "---------------------------"

    data_left = []
    data_right = []

    for i in range(rows):
        if data[i, attr] <= mean_split:
            data_left.append(data[i, :])
        else:
            data_right.append(data[i, :])

    data_left = np.array(data_left)
    if data_left.shape[0] != 0:
        data_left = np.delete(data_left, attr, axis=1)
    
    data_right = np.array(data_right)
    if data_right.shape[0] != 0:
        data_right = np.delete(data_right, attr, axis=1)

    if data_left.shape[0] != 0:
        run_split(node.leftNode, data_left)
    else:
        node.leftNode.isLeaf = True
        node.predictedVal = 0

    if data_right.shape[0] != 0:
        run_split(node.rightNode, data_right)
    else:
        node.rightNode.isLeaf = True
        node.predictedVal = 0

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

def printTree(node):
    if node.isLeaf:
        print "None, None, True, 0"
    else:
        print node.attrValue, node.mean_split, node.isLeaf, node.predictedVal
        printTree(node.leftNode)
        printTree(node.rightNode)

def predictor(node, r):
    if node.isLeaf:
        return node.predictedVal
    
    attr_value = r[node.attrValue]
    del_row = r
    del_row = np.delete(del_row, node.attrValue)

    if attr_value <= node.mean_split:
        return predictor(node.leftNode, del_row)
    return predictor(node.rightNode, del_row)

def run_decision_tree():
    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]

    data = np.array(list(data)).astype("float")
    # print data
    print "Number of records: %d" % len(data)

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    avg_accuracy = 0
    for j in range(K):
        training_set = [x for i, x in enumerate(data) if i % K != j]
        training_set = np.array(training_set)
        test_set = [x for i, x in enumerate(data) if i % K == j]
        test_set = np.array(test_set)

        tree = DecisionTree()
        # Construct a tree using training set
        tree.learn(training_set)

        root = Tree()

        run_split(root, training_set)
        # printTree(root)

        # Classify the test set using the tree we just constructed
        results = []
        for instance in test_set:
            result = predictor(root, instance[:-1])
            results.append( result == instance[-1])

        # Accuracy
        accuracy = float(results.count(True))/float(len(results))
        print "accuracy: %.4f" % accuracy 
        avg_accuracy += accuracy      
        
        # Writing results to a file (DO NOT CHANGE)
        f = open(myname+"result.txt", "a")
        # f.write(str(j + 1) + " epoch\n")
        # f.write("accuracy: %.4f" % accuracy)
        # f.write("\n")
    
    avg_accuracy /= K
    print "average accuracy: %.4f" % avg_accuracy
    f.write("average accuracy: %.4f" % avg_accuracy)
    f.write("\n")
    f.close()

if __name__ == "__main__":
    run_decision_tree()
