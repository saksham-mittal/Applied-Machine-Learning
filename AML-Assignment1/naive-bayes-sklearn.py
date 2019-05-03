import numpy as np
import csv
import json
from sklearn.preprocessing import StandardScaler

with open('train.json') as f:
    data_train = json.load(f)

with open('test.json') as f1:
    data_test = json.load(f1)

ingredient_list = []
labels = []
for elem in data_train:
    labels.append(elem.get("cuisine"))
    ingredient = elem.get("ingredients")
    for elem2 in ingredient:
        ingredient_list.append(elem2)

ingredient_list = list(set(ingredient_list))

# Converting labels to np array
labels = np.array(labels)

X_train = np.zeros((labels.shape[0], len(ingredient_list)))

row_num = 0
for elem in data_train:
    ingredient = elem.get("ingredients")
    for elem2 in ingredient:
        if elem2 in ingredient_list:
            index = ingredient_list.index(elem2)
            X_train[row_num, index] = 1
    row_num += 1

# np.savetxt('train2.csv', X_train, delimiter=',')

len_test = 0
for elem in data_test:
    len_test += 1

X_test = np.zeros((len_test, len(ingredient_list)))
row_num = 0
id_list = []
for elem in data_test:
    id_list.append(elem.get("id"))
    ingredient = elem.get("ingredients")
    for elem2 in ingredient:
        if elem2 in ingredient_list:
            index = ingredient_list.index(elem2)
            X_test[row_num, index] = 1
    row_num += 1

# np.savetxt('test2.csv', X_test, delimiter=',')

print "Data splitted and preprocessed"

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print "Scaling finished"

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, labels)
print "Train data fitted"

y_pred = classifier.predict(X_test)
print "Data predicted"

text_file = open("Output_naive_bayes.csv", "w")
text_file.write("id,cuisine\n")

count = 0
for elem in id_list:
    text_file.write("%d,%s\n" % (elem, y_pred[count]) )
    count += 1

text_file.close()
print "Data written to csv"
