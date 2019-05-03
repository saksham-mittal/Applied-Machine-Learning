import pandas as pd
import numpy as np

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

	# Finding test error on the following lambda values
	lambda_values = [0.01, 0.1, 1, 10, 100]
	min_test_mse = 100.0

	for l in lambda_values:
		w = mylinridgereg(training_set, training_labels, l)
		target_test = mylinridgeregeval(test_set, w)

		mse_test = meansquarederr(target_test, test_labels)
		if mse_test < min_test_mse:
			lambda_best_performance = l
			min_test_mse = mse_test
			weight_values_for_min_mse = w

		target_train = mylinridgeregeval(training_set, w)

		mse_train = meansquarederr(target_train, training_labels)
		print "For lambda({}), MSE train =".format(l), mse_train, "and MSE test =", mse_test

	print "Lambda with best performance =", lambda_best_performance

	abs_weight = np.absolute(weight_values_for_min_mse)
	sorted_weight = np.sort(abs_weight)
	ind1 = np.where(abs_weight==sorted_weight[0])[0][0]
	ind2 = np.where(abs_weight==sorted_weight[1])[0][0]
	# ind1 and ind2 are the features to be removed
	# because they have very less impact(weight) on the model

	weight_removed = np.delete(weight_values_for_min_mse, [ind1, ind2])

	# New target value for test set
	target_test = mylinridgeregeval(np.delete(test_set, [ind1, ind2], 1), weight_removed)

	mse_test = meansquarederr(target_test, test_labels)

	# New target value for train set
	target_train = mylinridgeregeval(np.delete(training_set, [ind1, ind2], 1), weight_removed)

	mse_train = meansquarederr(target_train, training_labels)
	print "After removing redundant features, MSE train =", mse_train, "and MSE test =", mse_test


if __name__ == "__main__":
    main()