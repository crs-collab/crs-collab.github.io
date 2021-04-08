import pickle
# Filter the uneccesary warnings
import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
# Author : Samyuktha G
warnings.filterwarnings("ignore")

distance_time_speed_dataset = pd.read_csv('distance_time_speed_dataset.csv')

s = (distance_time_speed_dataset.dtypes == 'object')
object_cols = list(s[s].index)
label_encoder = LabelEncoder() # encoding all Object columns
list_of_encoding = []
for col in object_cols:
    distance_time_speed_dataset[col] = label_encoder.fit_transform(distance_time_speed_dataset[col])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    list_of_encoding.append(le_name_mapping) # mapping the labels

z_scores = zscore(distance_time_speed_dataset)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1) # filtering rows that are three std deviations away from the mean
new_df = distance_time_speed_dataset[filtered_entries]

train, test = train_test_split(new_df, test_size= 0.3, random_state=42, shuffle=True)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

X_train = train.drop(['tracking_number', 'time_to_delivery_hours'], axis = 1)
Y_train = train['time_to_delivery_hours']
X_test  = test.drop(['tracking_number', 'time_to_delivery_hours'], axis = 1)
result = pd.DataFrame()
result['Actual'] = test['time_to_delivery_hours']

from sklearn.ensemble import RandomForestRegressor
# define the model
model = RandomForestRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
model.fit(X_train, Y_train)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train)
# Xt_scaled = scaler.transform(X_test)
# from sklearn.neural_network import MLPRegressor
# MLPregr = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_scaled, Y_train)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))