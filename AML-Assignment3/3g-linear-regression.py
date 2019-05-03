import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_data_f(split, mse_test_list, mse_train_list):
    plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
    plt.title("Effect of lambda on error change on different partition fraction values")
    plt.locator_params(axis='x', nbins=24)
    plt.xlabel("Lambda")
    plt.ylabel("MSE")

    x_labels = [x + 1 for x in range(len(mse_test_list))]
    plt.plot(x_labels, mse_test_list, color='#32CD32',
        marker='+', label="Test data MSE")
    
    x_labels = [x + 1 for x in range(len(mse_train_list))]
    plt.plot(x_labels, mse_train_list, color='red', marker='D',
        ms=3, label="Train data MSE")
    plt.legend()

    fileName = "plot-{}".format(split) + ".png"
    plt.savefig(fileName, bbox_inches='tight')

def plot_data_g(x_label_list, min_average_MSE_test_list, lmbda_list):
    plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
    plt.title("Minimum average test MSE versus the partition fraction values")
    plt.locator_params(axis='x', nbins=5)
    plt.xlabel("Split value")
    plt.ylabel("Minimum average test MSE")

    plt.plot(x_label_list, min_average_MSE_test_list, color='#32CD32', marker='+')

    fileName = "plot-g1.png"
    plt.savefig(fileName, bbox_inches='tight')

    plt.figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
    plt.title("Lambda that produced the minimum average test MSE versus the partition fraction values")
    plt.locator_params(axis='x', nbins=5)
    plt.xlabel("Split value")
    plt.ylabel("Lambda")

    plt.plot(x_label_list, lmbda_list, color='#32CD32', marker='+')

    fileName = "plot-g2.png"
    plt.savefig(fileName, bbox_inches='tight')

def split_data(data):
    split_arr = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_average_MSE_test_list = []
    lmbda_list = []

    for split_fraction in split_arr:
        training_set = data.sample(frac = split_fraction)
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

        # Find the average MSE for the current split and various lambdas
        lmbda = 1
        mse_test_list = []
        mse_train_list = []
        min_average_MSE_test = 10000
        l = 10000
        for i in range(25):
            w = mylinridgereg(training_set, training_labels.values, lmbda)

            target_test = mylinridgeregeval(test_set, w)

            mse_test = meansquarederr(target_test, test_labels.values)
            if mse_test < min_average_MSE_test:
                min_average_MSE_test = mse_test
                l = lmbda
            mse_test_list.append(mse_test)

            target_train = mylinridgeregeval(training_set, w)

            mse_train = meansquarederr(target_train, training_labels.values)
            mse_train_list.append(mse_train)
            print "For lambda({}), MSE train =".format(lmbda), mse_train, "and MSE test =", mse_test

            lmbda = lmbda + 1
        min_average_MSE_test_list.append(min_average_MSE_test)
        lmbda_list.append(l)

        plot_data_f(split_fraction, mse_test_list, mse_train_list)
    
    plot_data_g(split_arr, min_average_MSE_test_list, lmbda_list)

def main():
	# Reading data and adding column names for the dataframes
	data = pd.read_csv("linregdata", names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked Weight", "Vicera weight", "Shell weight", "Rings"])

	one_hot_encoding = pd.get_dummies(data['Sex'])

	# Dropping the sex column
	data = data.drop(labels='Sex', axis=1)

	# Concatenating the one hot encoding of Sex to data
	data = pd.concat([one_hot_encoding, data], axis=1)

	# Splitting data into test and train for different splits and linear ridge regression
	split_data(data)

if __name__ == "__main__":
    main()