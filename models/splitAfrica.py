import pandas as pd
import sys
import numpy as np
from scipy.misc import imshow, imresize
import imageio
import os


def load_df(povertyCSVPath):
    print("Loading data from {}".format(povertyCSVPath))
    df = pd.read_csv(povertyCSVPath).dropna()
    return df 
    
def extractCountry(df, lat, lon):
     cur = df.loc[(df['lon'] == lon) & (df['lat'] == lat)]
     if len(cur.index) > 0:
         return cur["country"].iloc[0]
     return None



if __name__ == "__main__":

    rootDir = "/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/"
    povertyDataset = rootDir + "data/wealth_index_cluster_locations_2017_08.csv"
    africaData = "/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/AfricaArticleClusters.npy"

    allData = np.load(africaData)
    df = load_df(povertyDataset) 
    print("Loaded all data") 
    
    newData = []
    cnt = 0 
    for row in allData: 
        country = extractCountry(df, row[0][0], row[0][1])
        if country == None:
            continue 

        print("Cnt: {}".format(cnt))
        cnt += 1
        print("Extracting country for lat {}, lon {} with result {}".format(str(row[0][0]), str(row[0][1]), country))
        tmp = list(row) 
        tmp.append(country) 
        newData.append(tmp) 
    np.save("AfricaDataWithCountry.npy", newData)
