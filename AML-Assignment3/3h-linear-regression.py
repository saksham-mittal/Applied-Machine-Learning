"""
Usage: python 3h-linear-regression.py 0.7 9
"""
import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import sys

# X and Y are dataframes
def mylinridgereg(X, Y, lmbda):
    temp1 = np.dot(X.T, X)
    temp2 = np.identity(X.shape[1]) * lmbda
    temp = np.linalg.inv(temp1 + temp2)
    temp = np.dot(temp, np.dot(X.T, Y))
    return temp

def mylinridgeregeval(X, weights):
    return np.dot(X, weights)

def meansquarederr(T, Tdash):
    return (np.square(T - Tdash)).mean(axis=None)

def plot_data(x1, y1, x2, y2):
    plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
    plt.title("Predicted versus Actual label values for Test Data")
    plt.xlabel("Predicted value")
    plt.ylabel("Actual labels")
    plt.scatter(x1, y1, marker='D', s=10)
    x = np.arange(25)
    plt.gca().set_prop_cycle("color", ['#32CD32'])
    plt.plot(x, x)
    fileName = "plot-h1.png"
    plt.savefig(fileName, bbox_inches='tight')

    plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
    plt.title("Predicted versus Actual label values for Train Data")
    plt.xlabel("Predicted value")
    plt.ylabel("Actual labels")
    plt.scatter(x2, y2, marker='D', s=10)
    plt.gca().set_prop_cycle("color", ['#32CD32'])
    plt.plot(x, x)
    fileName = "plot-h2.png"
    plt.savefig(fileName, bbox_inches='tight')

def main(split_fraction, lmbda):
	# Reading data and adding column names for the dataframes
    data = pd.read_csv("linregdata", names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked Weight", "Vicera weight", "Shell weight", "Rings"])

    one_hot_encoding = pd.get_dummies(data['Sex'])

	# Dropping the sex column
    data = data.drop(labels='Sex', axis=1)

	# Concatenating the one hot encoding of Sex to data
    data = pd.concat([one_hot_encoding, data], axis=1)

    # Splitting the data into test and train
    training_set = data.sample(frac = 0.8)
    test_set = data.drop(training_set.index)

    # Storing the labels separately and dropping them
    training_labels = training_set["Rings"]
    del training_set['Rings']

    test_labels = test_set["Rings"]
    del test_set["Rings"]

    # Normalizing the data
    training_set = (training_set - training_set.mean()) / training_set.std()
    test_set = (test_set - test_set.mean()) / test_set.std()

    training_set = training_set.values
    test_set = test_set.values

    training_set = np.append(np.ones((training_set.shape[0], 1), dtype=int), training_set, axis=1)
    test_set = np.append(np.ones((test_set.shape[0], 1), dtype=int), test_set, axis=1)

    # Finding test error on the best lambda value found in part(g)
    w = mylinridgereg(training_set, training_labels.values, lmbda)

    target_train = mylinridgeregeval(training_set, w)

    target_test = mylinridgeregeval(test_set, w)

    mse_test = meansquarederr(target_test, test_labels.values)
    print mse_test

    plot_data(test_labels.values, target_test, training_labels.values, target_train)

if __name__ == "__main__":
    split_fraction = float(sys.argv[1])
    lmbda = float(sys.argv[2])
    main(split_fraction, lmbda)