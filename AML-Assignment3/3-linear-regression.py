import pandas as pd
import numpy as np

# X and Y are dataframes
def mylinridgereg(X, Y, lmbda):
    X = X.values
    Y = Y.values
    temp1 = np.dot(X.T, X)
    temp2 = np.identity(X.shape[1]) * lmbda
    temp = np.linalg.inv(temp1 + temp2)
    temp = np.dot(temp, np.dot(X.T, Y))
    return temp

def mylinridgeregeval(X, weights):
    return np.dot(X, weights)

def meansquarederr(T, Tdash):
    return (np.square(T - Tdash)).mean(axis=None)

def main():
    # Reading data and adding column names for the dataframes
	data = pd.read_csv("linregdata", names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked Weight", "Vicera weight", "Shell weight", "Rings"])
	# print data

	# One hot encoding the Sex column
	one_hot_encoding = pd.get_dummies(data['Sex'])

	# Dropping the sex column
	data = data.drop(labels='Sex', axis=1)

	# Concatenating the one hot encoding of Sex to data
	data = pd.concat([one_hot_encoding, data], axis=1)

	# Splitting data into test and train
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

if __name__ == "__main__":
    main()