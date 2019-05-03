import pandas as pd
import numpy as np
import datetime as dt

def deg2rad(deg):
    return deg * (np.pi / 180)
  
def getDistanceFromLatLongInKm(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)
    a = np.sin(dLat / 2)**2 + np.cos(deg2rad(lat1)) * np.cos(deg2rad(lat2)) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d

training_set = pd.read_csv("train.csv", nrows=1000000)

training_set = training_set.dropna(how = 'any', axis = 'rows')

# Adding an attribute 'distance
training_set['distance'] = getDistanceFromLatLongInKm(training_set['pickup_latitude'], training_set['pickup_longitude'], training_set['dropoff_latitude'], training_set['dropoff_longitude'])

# Dropping the latitude and longitude attributes
training_set = training_set.drop(labels=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], axis=1)

# Deleting the key attribute
del training_set['key']

training_set['pickup_datetime_dt'] = training_set['pickup_datetime'].apply(lambda d: dt.datetime.strptime(d, '%Y-%m-%d %H:%M:%S UTC'))

del training_set['pickup_datetime']

datetime_temp = []
for elem in training_set['pickup_datetime_dt']:
    datetime_temp.append(elem.year)
    
training_set['year'] = np.array(datetime_temp)

datetime_temp = []
for elem in training_set['pickup_datetime_dt']:
    datetime_temp.append(elem.month)
    
training_set['month'] = np.array(datetime_temp)

datetime_temp = []
for elem in training_set['pickup_datetime_dt']:
    datetime_temp.append(elem.day)
    
training_set['day'] = np.array(datetime_temp)

datetime_temp = []
for elem in training_set['pickup_datetime_dt']:
    datetime_temp.append(elem.hour)
    
training_set['hour'] = np.array(datetime_temp)

datetime_temp = []
for elem in training_set['pickup_datetime_dt']:
    datetime_temp.append(elem.minute)
    
training_set['minute'] = np.array(datetime_temp)

del training_set['pickup_datetime_dt']

training_labels = training_set['fare_amount']

del training_set['fare_amount']

print(training_set)

# Normalizing the training_set

test_set = pd.read_csv("test.csv")

# Adding an attribute 'distance
test_set['distance'] = getDistanceFromLatLongInKm(test_set['pickup_latitude'], test_set['pickup_longitude'], test_set['dropoff_latitude'], test_set['dropoff_longitude'])

key_values = test_set['key']

# Deleting the key attribute
del test_set['key']

# Dropping the latitude and longitude attributes
test_set = test_set.drop(labels=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], axis=1)

test_set['pickup_datetime_dt'] = test_set['pickup_datetime'].apply(lambda d: dt.datetime.strptime(d, '%Y-%m-%d %H:%M:%S UTC'))

del test_set['pickup_datetime']

datetime_temp = []
for elem in test_set['pickup_datetime_dt']:
    datetime_temp.append(elem.year)
    
test_set['year'] = np.array(datetime_temp)

datetime_temp = []
for elem in test_set['pickup_datetime_dt']:
    datetime_temp.append(elem.month)
    
test_set['month'] = np.array(datetime_temp)

datetime_temp = []
for elem in test_set['pickup_datetime_dt']:
    datetime_temp.append(elem.day)
    
test_set['day'] = np.array(datetime_temp)

datetime_temp = []
for elem in test_set['pickup_datetime_dt']:
    datetime_temp.append(elem.hour)
    
test_set['hour'] = np.array(datetime_temp)

datetime_temp = []
for elem in test_set['pickup_datetime_dt']:
    datetime_temp.append(elem.minute)
    
test_set['minute'] = np.array(datetime_temp)

del test_set['pickup_datetime_dt']

print(test_set)

# Code for Random forests regression
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=50)
reg.fit(training_set, training_labels)

ans1 = reg.predict(test_set)

import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(training_set, training_labels)

ans2 = xgb.predict(test_set)

# Writing ans1 to file
f = open("output-random-forests.csv", "w")
f.write("key,fare_amount\n")

for i in range(ans1.shape[0]):
    f.write("%s,%.2f\n" % (key_values[i],ans1[i]))

# Writing ans2 to file
f = open("output-xgboost.csv", "w")
f.write("key,fare_amount\n")

for i in range(ans2.shape[0]):
    f.write("%s,%.2f\n" % (key_values[i],ans2[i]))