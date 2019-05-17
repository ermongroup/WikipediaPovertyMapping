import pandas as pd
import sys
import numpy as np
from scipy.misc import imshow, imresize
import gdal
from utils.geotiling import *
import imageio

def visualize_location(props, gdal_tif, center_lon, center_lat, image_pixel_size=40000, 
                    downSize=500, verbose=False):
    '''
    Reads the image_pixel_size by image_pixel_size image at the longitude center_lon and latitude center_lat
    Modified from implementation by Anthony Perez
    '''

    center_col, center_row = props.lonlat2colrow(center_lon, center_lat)
    left_col, top_row = center_col - image_pixel_size // 2, center_row - image_pixel_size // 2

    # No input checking so this call may return an error (give it a try, use center_lon = 100000.0)
    image = gdal_tif.ReadAsArray(left_col, top_row, image_pixel_size, image_pixel_size)
    if len(image.shape) == 2:
        image = imresize(image, (downSize, downSize))
        if verbose:
            print("Image shape: {}".format(image.shape))
        return image

    image = np.transpose(image, (1,2,0))
    if verbose:
        print("Image shape: {}".format(image.shape))

    return image


def printInfo(filepath):
    """
    Prints some information regarding the gdal file 
    """
    info = gdal.Info(filepath, deserialize=True)
    print(info)

def pdHandler(pdRow, props, gdal_tif, image_pixel_size, source_tif, outputBase,  downSize, verbose=True):

    center_lon = pdRow["lon"]
    center_lat = pdRow["lat"]
    image = visualize_location(props, gdal_tif, center_lon, center_lat, image_pixel_size=image_pixel_size, 
                    downSize=downSize, verbose=verbose)

    outputFile = outputBase + str(center_lon) + "," + str(center_lat) + ".npy"
    if verbose:
        print("Saving file: {}".format(outputFile))
    np.save(outputFile, image)
#    imageio.imwrite(root + "output/" + str(center_lon) + "," + str(center_lat) + ".png", image) 

def generateDataset(povertyDataset, gufDataset, countryName, outputPath, 
        image_pixel_size=40000, downSize=500):

    # Read the file and load the meta data
    props = GeoProps()
    gdal_tif = gdal.Open(gufDataset)
    props.import_geogdal(gdal_tif)

    povData = pd.read_csv(povertyDataset).dropna()
    count = povData['country'].str.contains(countryName).sum() 
    print("Generating {} images".format(count))
    if count > 0:
        countryOnly = povData.loc[povData['country'] == country]
        countryOnly.apply(pdHandler, axis=1, raw=True, args=(props, gdal_tif, image_pixel_size, gufDataset, outputPath, downSize))

    print("Dataset finished generating")


if __name__ == "__main__":

    rootDir = "/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/"
    povertyDataset = rootDir + "data/wealth_index_cluster_locations_2017_08.csv"
    gufDataset = rootDir + "data/GUF_Continent_Africa.tif"
    outputPath = rootDir + "data/UgandaGUF/"
    country = "Uganda" 

#    print("Merging datasets: {}, {}".format(povertyDataset, gufDataset))
#    printInfo(gufDataset)

    generateDataset(povertyDataset, gufDataset, country, outputPath) 

