import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from opencage.geocoder import OpenCageGeocode  # API to convert places to coordinates
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from vincenty import vincenty  # Calculate the geographical distance (in kilometers) between 2 points
# Author : Samyuktha G
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])

def predict():
    distance_time_speed_dataset = pd.read_csv('distance_time_speed_dataset.csv')
    s = (distance_time_speed_dataset.dtypes == 'object')
    object_cols = list(s[s].index)
    label_encoder = LabelEncoder()  # encoding all Object columns
    list_of_encoding = []
    for col in object_cols:
        distance_time_speed_dataset[col] = label_encoder.fit_transform(distance_time_speed_dataset[col])
        le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        list_of_encoding.append(le_name_mapping)  # mapping the labels
    z_scores = zscore(distance_time_speed_dataset)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)  # filtering rows that are three std deviations away from the mean
    new_df = distance_time_speed_dataset[filtered_entries]

    distribution = []
    carrier = new_df['carrier']
    counter = Counter(carrier)
    for key, value in counter.items():
        percent = value / len(carrier) * 100
        df = {'Carrier': key, 'Count of unique values': value, 'Percentage': percent}
        distribution.append(df)
    dist = pd.DataFrame(distribution)
    dist = dist.sort_values(by=['Carrier'], ignore_index=True)

    group = new_df.groupby('carrier')
    df1 = group.apply(lambda x: len(x['source_city'].unique()))
    df2 = group.apply(lambda x: len(x['destination_city'].unique()))
    df3 = group.apply(lambda x: sum(x['time_to_delivery_hours']) / len(x['time_to_delivery_hours']))
    df4 = group.apply(lambda x: sum(x['speed']) / len(x['speed']))
    df5 = group.apply(lambda x: sum(x['distance']) / len(x['distance']))

    dist['Unique source cities'] = df1.values
    dist['Unique destination cities'] = df2.values
    dist['Average delivery time'] = df3.values
    dist['Average speed'] = df4.values
    dist['Average distance'] = df5.values
    key = '7d84bfd479184b6fb58f0c46bfc4debc'  # API Key
    geocoder = OpenCageGeocode(key)
    src = request.form.get("sourceCity", False)
    dest = request.form.get("destinationCity", False)
    src_location = geocoder.geocode(src)
    if src_location and len(src_location):
        src_longitude = src_location[0]['geometry']['lng']
        src_latitude = src_location[0]['geometry']['lat']
        src_code = src_location[0]['components']['country_code']
    else:
        return render_template('index.html',pred='Please enter correct source location')
    dest_location = geocoder.geocode(dest)
    if dest_location and len(dest_location):
        dest_longitude = dest_location[0]['geometry']['lng']
        dest_latitude = dest_location[0]['geometry']['lat']
        dest_code = dest_location[0]['components']['country_code']
    else:
        return render_template('index.html',pred='Please enter correct destination location')

    distance = vincenty((src_latitude, src_longitude), (dest_latitude, dest_longitude))

    srcC, destC = 100, 100
    srcS, destD = 396, 2207

    if src in list_of_encoding[0]:
        srcS = list_of_encoding[0].get(src)

    if dest in list_of_encoding[1]:
        destD = list_of_encoding[1].get(dest)

    if src_code in list_of_encoding[2]:
        srcC = list_of_encoding[2].get(src_code)

    if dest_code in list_of_encoding[3]:
        destC = list_of_encoding[3].get(dest_code)

    carriers = ['Aramex', 'BlueDart', 'Delhivery', 'DHL Express', 'dotzot', 'Ecom Express', 'FedEx', 'XpressBees']

    min_value = math.inf
    best_carrier = ""
    for id in range(8):
        item = [[srcS, destD, id, distance, srcC, destC, dist["Average speed"][id]]]
        value = model.predict(item)
        if value < min_value and value>0:
            min_value = value
            best_carrier = carriers[id]

    answer =  best_carrier + " is predicted to deliver the fastest!"
    #answer = best_carrier + "is predicted to deliver fastest in" + min_value[0][0] + "hours /" + min_value[0][0] // 24 + "days"
    return render_template('index.html', pred = answer)

if __name__ == '__main__':
    app.run(debug=True)