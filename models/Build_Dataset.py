import sys
sys.path.append("/atlas/u/esheehan/wikipedia_project/dataset/text_dataset/dataset_modules")
#from data_processor import *
import numpy as np
import gensim
import pandas as pd
import pickle

import math
from math import sin, cos, sqrt, atan2, radians
import tensorflow as tf
import collections
import os

PATH = "/atlas/u/esheehan/wikipedia_project/RNN/Doc2Vec/models/wikimodel_DBOW_vector300_window8_count15_epoch10/wikimodel.doc2vec"

# The distance in km to check within
MARGIN = 10

# The number of km in one degree of latitude
LAT_KM = 110.574

# The number of km in one degree of longitude
LON_KM = 111.320


# Returns the embeddings and the average embedding of an array of articles
def get_embeddings_average(array, model):
    
    if len(array) == 0:
        return [], None

    embeddings = []
    for i in array:
        embeddings.append(model.docvecs[get_title(i[1])])

    av = []
    for i in range(len(embeddings[0])):
        sum_ = 0
        for j in embeddings:
            sum_ += j[i]
        sum_ /= float(len(embeddings))
        av.append(sum_)

    return embeddings, av

# Given coordinates a, b in deg, return the distance between a and b in km         
def compute_distance(c1, c2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(c1[0])
    lon1 = radians(c1[1])
    lat2 = radians(c2[0])
    lon2 = radians(c2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def _parse_function(example_proto):
    features = {
        band_name : tf.FixedLenFeature(shape=[255**2], dtype=tf.float32) for band_name in ['LAT','LON','RED','GREEN','BLUE','NIGHTLIGHTS']
    }
    features['wealthpooled'] = tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    parsed_features = tf.parse_single_example(example_proto, features)
    for band_name in ['RED','GREEN','BLUE']:
        parsed_features[band_name] = tf.cast(parsed_features[band_name], tf.float64)
    parsed_features['wealthpooled'] = tf.cast(parsed_features['wealthpooled'], tf.float64)
    parsed_features['LAT'] = tf.cast(parsed_features['LAT'], tf.float64)
    parsed_features['LON'] = tf.cast(parsed_features['LON'], tf.float64)
    return parsed_features 

def find_closest_article(lat, lon, wealth, country, array, model):

    distances = []
    # Find the distance to all articles
    for i in array:
        curDist = compute_distance([lat,lon], [i[3][0], i[3][1]])
        distances.append((curDist, i))
    distances.sort()

    # Get all articles within 10km
    ind = 0
    while ind < len(distances) and distances[ind][0] <= MARGIN:
        ind += 1
    if ind < len(distances):
        within_margin, av1 = get_embeddings_average(distances[:ind], model)
        # Distance, title, coordinates, category, embedding
        within_m = [[distances[i][0], get_title(distances[i][1]), get_coordinates(distances[i][1]),\
                get_category(distances[i][1]), within_margin[i]] for i in range(ind)]
    else:
        within_m, av1 = [], None

    # Get the nearest 10 articles
    nearest_10, av2 = get_embeddings_average(distances[:10], model)
    n_10 = [[distances[i][0], get_title(distances[i][1]), get_coordinates(distances[i][1]),\
            get_category(distances[i][1]), nearest_10[i]] for i in range(10)]

    print("Obtained " + str(ind) + " articles within 10km of " + str(lat) + ", " + str(lon))
    return[(lat, lon), country, wealth, av1, within_m, av2, n_10]
    
def process_record(record, array, model, sess):

    country = record.split("_")[3]
    dataset = tf.data.TFRecordDataset("/atlas/u/chenlin/wikipedia_images/Poverty_prediction/original_tfrecord/"  +record, compression_type="GZIP")
    image = dataset.map(_parse_function)
    image = image.batch(1)
    iterator_image = image.make_one_shot_iterator()
    next_image = iterator_image.get_next()
    data = []
    i = 0
    while True:
        try:
            Image = sess.run(next_image)
        except:
            break

        print(str(i))
        i += 1

        lat = float(np.mean(np.array(Image['LAT']),axis=1))
        lon = float(np.mean(np.array(Image['LON']),axis=1))
        
        r = Image['RED']
        r = np.array(r).reshape(255, 255)
        g = Image['GREEN']
        g = np.array(g).reshape(255, 255)
        b = Image['BLUE']
        b = np.array(b).reshape(255, 255)
        n = Image['NIGHTLIGHTS']
        n = np.array(n).reshape(255, 255)
        ar =  np.stack([r, g, b, n])
        path = "Poverty_Images/" + country
        print(path)

        # Save the images
        if not os.path.isdir("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/" + path):
            os.mkdir("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/" + path)
        np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/" + path + "/" + str(lat) + "_" + str(lon), ar)

        # Get the nearest article data and save it with the path to the image
        art = find_closest_article(lat, lon, Image['wealthpooled'], country, array, model)
        art.append("Poverty_Images/" + country + "/" + str(lat) + "_" + str(lon))
        data.append(art)

    return data

"""if __name__ == "__main__":
    
    array = load_coordinate_array("full", uncategorized=True, verbose=True)
    model = gensim.models.Doc2Vec.load(PATH)
    print("Loaded Doc2Vec model...")
    lat1 = -2.0
    lon1 = 20.0
    lat2 = 38.6727
    lon2 = -19.2435
    array = get_articles_within_margin(array, (lat1, lon1), point=(lat2, lon2), verbose=True, map="merc", display=True)
    print("Loaded all articles...")
    

    data = []
    sess = tf.Session()

    if not os.path.isdir("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Poverty_Images"):
        os.mkdir("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Poverty_Images")
    

    baseDir = "/atlas/u/chenlin/wikipedia_images/Poverty_prediction/original_tfrecord"
    for record in os.listdir(baseDir):
        print(str(record))
        data += process_record(record, array, model, sess)

    print("Obtained data for " + str(len(data)) + " clusters!") 
    np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Full_Clusters", data)
    print("Data saved!")"""

def add_record(record, sess):
    country = record.split("_")[3]
    year = record.split("_")[2]
    year = int(year[:4])
    dataset = tf.data.TFRecordDataset("/atlas/u/chenlin/wikipedia_images/Poverty_prediction/original_tfrecord/"  +record, compression_type="GZIP")
    image = dataset.map(_parse_function)
    image = image.batch(1)
    iterator_image = image.make_one_shot_iterator()
    next_image = iterator_image.get_next()
    data = []
    i = 0
    while True:
        try:
            Image = sess.run(next_image)
        except:
            break

        print(str(i), year)
        i += 1

        lat = float(np.mean(np.array(Image['LAT']),axis=1))
        lon = float(np.mean(np.array(Image['LON']),axis=1))
        wealth = Image['wealthpooled']
        data.append([country, (lat, lon), wealth, year])
    return data

if __name__ == "__main__":

    sess = tf.Session()
    data = []
    clusters_temp = np.load("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Full_Clusters.npy")

    baseDir = "/atlas/u/chenlin/wikipedia_images/Poverty_prediction/original_tfrecord"
    for record in os.listdir(baseDir):
        print(str(record))
        data += add_record(record, sess)

    clusters = []
    for i in clusters_temp:
        point = None
        for j in data:
            if j[0] == i[1] and i[0][0] == j[1][0] and i[0][1] == j[1][1] and i[2] == j[2]:
                point = list(i)
                point.append(j[3])
                break
        if point is None:
            print("Skipped " + str(i[0]))
        else:
            clusters.append(point)

    np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Full_Clusters_Updated", clusters)



