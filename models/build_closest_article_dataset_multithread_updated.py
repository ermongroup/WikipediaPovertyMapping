
import sys
sys.path.append("/atlas/u/esheehan/wikipedia_project/dataset/text_dataset/dataset_modules")
from data_processor import *
import numpy as np
import gensim
import pandas as pd
import pickle
import random
import math
from math import sin, cos, sqrt, atan2, radians

PATH = "/atlas/u/esheehan/wikipedia_project/RNN/Doc2Vec/models/wikimodel_DBOW_vector300_window8_count15_epoch10/wikimodel.doc2vec"

# The distance in km to check within
MARGIN = 10

# The number of km in one degree of latitude
LAT_KM = 110.574

# The number of km in one degree of longFFFFFFitude
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

def find_closest_article(row, array, model, lat1, lon1, lat2, lon2, lat_range, lon_range):
    lat = float(row["lat"])
    lon = float(row["lon"])
    # If the coordinate lies within Africa
    if row["iso3"] == "BGD":#lat <= (lat1 + lat_range) and lat >= (lat1 - lat_range) and lon <= (lon1 + lon_range) and lon >= (lon1 - lon_range):
        wealth = float(row["wealthpooled"])
        country = row["iso3"]
        distances = []
        # Find the distance to all articles
        for i in array:
            #delta_y = i[3][0] - lat
            #delta_x = i[3][1] - lon

            # Convert to km
            #delta_y *= LAT_KM
            #delta_x *= LON_KM * math.cos((math.pi / 180 ) * lat) 
            # distances.append([math.sqrt(delta_y**2 + delta_x**2), i])
            # print(math.sqrt(delta_y**2 + delta_x**2))
            curDist = compute_distance([lat,lon], [i[3][0], i[3][1]])
            distances.append((curDist, i))
            # print(curDist)
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
        #n_all = [[distances[i][0], get_title(distances[i][1]), get_coordinates(distances[i][1]),\
                #get_category(distances[i][1]), model.docvecs[get_title(distances[i][1])]] for i in range(len(distances))]


        print("Obtained " + str(ind) + " articles within 10km of " + str(lat) + ", " + str(lon))
        ret =  [(lat, lon), country, wealth, av1, within_m, av2, n_10]
        #full = [(lat, lon), country, wealth, av1, within_m, av2, n_all]

        #return (ret, full)
        return ret
    return None
            
def find_closest_article_Philippines(lat, lon, array, model):

    distances = []
    # Find the distance to all articles
    for i in array:
            #delta_y = i[3][0] - lat
            #delta_x = i[3][1] - lon

            # Convert to km
            #delta_y *= LAT_KM
            #delta_x *= LON_KM * math.cos((math.pi / 180 ) * lat) 
            # distances.append([math.sqrt(delta_y**2 + delta_x**2), i])
            # print(math.sqrt(delta_y**2 + delta_x**2))
        curDist = compute_distance([lat,lon], [i[3][0], i[3][1]])
        distances.append((curDist, i))
            # print(curDist)
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
        #n_all = [[distances[i][0], get_title(distances[i][1]), get_coordinates(distances[i][1]),\
                #get_category(distances[i][1]), model.docvecs[get_title(distances[i][1])]] for i in range(len(distances))]


    print("Obtained " + str(ind) + " articles within 10km of " + str(lat) + ", " + str(lon))
    ret =  [(lat, lon), "philippines", None, av1, within_m, av2, n_10]
        #full = [(lat, lon), country, wealth, av1, within_m, av2, n_all]

        #return (ret, full)
    return ret

if __name__ == "__main__":
    
    array = load_coordinate_array("full", uncategorized=True, verbose=True)
    model = gensim.models.Doc2Vec.load(PATH)
    print("Loaded Doc2Vec model...")
    lat1 = 11.3760
    lon1 = 123.2480
    lat2 = 19.2939
    lon2 = 119.3369

    #lat1 = 24.0580
    #lon1 = 89.9348
    #lat2 = 27.5389
    #lon2 = 93.7361
    #lat1 = -16.4168
    #lon1 = -64.7432
    #lat2 = -5.9130
    #lon2 = -52.2628
    #lat1 = -2.0
    #lon1 = 20.0
    #lat2 = 38.6727
    #lon2 = -19.2435
    array = get_articles_within_margin(array, (lat1, lon1), point=(lat2, lon2), verbose=True, map="merc", display=True)
    print("Loaded all articles...")

    #df =  pd.read_csv("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/wealth_index_cluster_locations_2017_08.csv")

    # The ranges we are interested in
    lat_range = abs(lat1 - lat2)
    lon_range = abs(lon1 - lon2)

    data = []
    #all_ = []

    """for index, row in df.iterrows():
        cur = find_closest_article(row, array, model, lat1, lon1, lat2, lon2, lat_range, lon_range)
        print(index)

        if cur is not None:
            data.append(cur)
            #all_.append(cur1)"""
    coords = [(random.uniform(5, 19), random.uniform(115, 125)) for i in range(10000)]
    for i in coords:
        cur = find_closest_article_Philippines(i[0], i[1], array, model)
        if cur is not None:
            data.append(cur)
            #all_.append(cur1)


    print("Obtained data for " + str(len(data)) + " clusters!") 
    np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/PhilippinesArticleClustersFull", data)
    #np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/AfricaArticleClustersUpdated", data)
    #np.save("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/AfricaArticleClustersUpdatedFull", all_)
    print("Data saved!")


