import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
import collections
import os


# x label
def _parse_function(example_proto):
    features = {
        band_name : tf.FixedLenFeature(shape=[255**2], dtype=tf.float32) for band_name in ['LAT','LON','RED','GREEN','BLUE','NIGHTLIGHTS']
    }
    features['wealthpooled'] = tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    parsed_features = tf.parse_single_example(example_proto, features)
    for band_name in band_names:
        parsed_features[band_name] = tf.cast(parsed_features[band_name], tf.float64)
    parsed_features['wealthpooled'] = tf.cast(parsed_features['wealthpooled'], tf.float64)
    parsed_features['LAT'] = tf.cast(parsed_features['LAT'], tf.float64)
    parsed_features['LON'] = tf.cast(parsed_features['LON'], tf.float64)
    return parsed_features 

def find_embedding(doc2vecs, lat,lon):

    closest = float('inf')
    vec = None
    for i in range(len(doc2vecs)):
        if len(doc2vecs[i])>=5:
            temp = (lat-doc2vecs[i][0][0])**2 + (lon-doc2vecs[i][0][1])**2
            if temp<=closest:
                closest = temp
                vec = i
    return vec

def process_record(record, data):

    dataset = tf.data.TFRecordDataset(record, compression_type="GZIP")
    images = dataset.map(_parse_function)


    for image in images:

        lat = np.mean(np.array(image['LAT']),axis=1)
        lon = np.mean(np.array(image['LON']),axis=1)

        index = find_embedding(data, lat, lon)
        if index in used:
            print(str(lat), str(lon))
        
        r = Image['RED']
        r = np.array(r).reshape(255, 255)
        g = Image['GREEN']
        g = np.array(g).reshape(255, 255)
        b = Image['BLUE']
        b = np.array(b).reshape(255, 255)
        n = Image['NIGHTLIGHTS']
        n = np.array(n).reshape(255, 255))
        ar =  np.stack([r, g, b, n])
        path = "Poverty_Images/" + data[index][1]

        if not os.isfile("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/" + path):
            os.mkdir("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/" + path)
        np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/" + path + "/" + str(lat) + "_" + str(lon), ar)

if __name__ == "__main__":
    
    if not os.isfile("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Poverty_Images"):
        os.mkdir("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/Poverty_Images")

    baseDir = "/atlas/u/chenlin/wikipedia_images/Poverty_prediction/original_tfrecord"
    data = list(np.load("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/AfricaArticleClustersUpdated.npy"))

    for record in os.listdir(baseDir):
        process_record(record, data)


