import pandas as pd
import sys
import numpy as np
from scipy.misc import imshow, imresize
import imageio
import os

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import math
import scipy.stats as ss

num_articles = 1

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_average(i, num):

    embeddings = [j[4] for j in i[6][:num]]

    av = []
    for k in range(len(embeddings[0])):
        sum_ = 0
        for j in embeddings:
            sum_ += j[k]
        sum_ /= float(len(embeddings))
        av.append(sum_)
    return av

def concatenate(i, num):
    embeddings = []
    for index, article in enumerate(i[6]):
        embeddings += list(article[4]) + [article[0]]
        if index == num - 1:
            break
    return embeddings
# Images are 500x500 numpy arrays 
def load_model(num_articles):
    input_shape = 300 * num_articles + num_articles
    np.random.seed(1234)
    concat = True
    model = Sequential()
    if concat:
        model.add(Dense(512, input_shape=(input_shape,), kernel_initializer='normal', activation='sigmoid'))
    else:
        model.add(Dense(512, input_shape=(300,), kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(256, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(32, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination, 'mae'])
    print(model.summary())
    return model

# Each element in data is comprised of a list with:
# 0. (lat, lon) 1. country tag 2. wealth pooled data 3. average embedding within 5km radius 4. list of article
# embeddings within 5km radius 5. average embedding of 10 nearest articles 6.list of 10 closest articles
#
# The article lists contain: 0. distance to the cluster 1. article title 2. coordinates 3. category 4. embedding
def load_data(data, train, test, concat, clean_split, num=10):
    
    print(len(data))
    print("Trained on ", train)
    print("Tested on ", test)
    print("Number of closest articles: ", num)
    if train == test:
        if not clean_split:
            X,Y = [], []
            for i in data:
                if i[1] == train or (train == "africa" and len(i) == 7):
                    if not num == 10:
                        X.append((concatenate(i, num) if concat else get_average(i, num)))
                    else:
                        X.append((concatenate(i, num) if concat else i[5]))
                    Y.append(i[2])
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1234, shuffle=True)
        
        else:

            X,Y = [], []
            for i in data:
                if i[1] == train or (train == "africa" and len(i) == 7):
                    if not num == 10:
                        X.append(((concatenate(i, num) if concat else get_average(i, num)), [i[6][0]]))
                    else:
                        X.append(((concatenate(i, num) if concat else i[5]), [i[6][0]]))
                    Y.append(i[2])
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1234, shuffle=True)

            seen = set()
            for i in X_train:
                for j in i[1]:
                    seen.add(j[1])
            
            X_test_new = []
            Y_test_new = []
            for i, j in enumerate(X_test):
                split = False
                for k in j[1]:
                    if k[1] in seen:
                        split = True
                        break
                if not split:
                    X_test_new.append(j[0])
                    Y_test_new.append(Y_test[i])
            X_test = X_test_new
            Y_test = Y_test_new
            X_train = [i[0] for i in X_train]

    else:

        X_train, Y_train, X_test, Y_test = [], [], [], []

        for i in data:
            if len(i) == 7:
                if i[1] == train or (train == "africa" and not test  == i[1]):
                    if not num == 10:
                        X_train.append((concatenate(i, num) if concat else get_average(i, num)))
                    else:
                        X_train.append((concatenate(i, num) if concat else i[5]))
                    Y_train.append(i[2])
                if i[1] == test or (test == "africa" and not i[1] == train):
                    if not num == 10:
                        X_test.append((concatenate(i, num) if concat else get_average(i, num)))
                    else:
                        X_test.append((concatenate(i, num) if concat else i[5]))
                    Y_test.append(i[2])

        X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.0, random_state=1234, shuffle=True)
        X_test, _, Y_test, _ = train_test_split(X_test, Y_test, test_size=0.0, random_state=1234, shuffle=True)
        
    X_train, X_test, Y_train, Y_test = np.asarray(X_train), np.asarray(X_test), np.asarray(Y_train), np.asarray(Y_test)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)
    print("Y_train Shape: ", Y_train.shape)
    print("Y_test Shape: ", Y_test.shape)

    print("Data Loaded") 
    return X_train, X_test, Y_train, Y_test

def inter():
    global num_articles
    return load_model(num_articles)

def run_model(train, test, epochs, concat, clean_split, outputPath, weightsName, num, X_train, X_test, Y_train, Y_test):

    reg = KerasRegressor(build_fn=inter, epochs=epochs, verbose=1,validation_split=0.0)

    # kfold = KFold(n_splits=5, random_state=1234)
    # results = np.sqrt(-1*cross_val_score(reg, X_train, Y_train, scoring= "neg_mean_squared_error", cv=kfold))
    # print("Training RMSE mean and std from CV: {} {}".format(results.mean(),results.std()))

    print("Testing model")
    reg.fit(X_train, Y_train)
    prediction=reg.predict(X_test)
    print("R2: ", r2_score(Y_test, prediction))
    p = pearsonr(Y_test, prediction)[0]
    if p < 0:
        p = -p**2
    else:
        p = p**2
    print("Pearson's r: ", p)
    s = spearmanr(Y_test, prediction)[0]
    if s < 0:
        s = -s**2
    else:
        s = s**2

    print("Spearman's rank correlation rho^2 and p: ", s)
    pred_rank = ss.rankdata(prediction)
    true_rank = ss.rankdata(Y_test)
    meanDiff = np.mean(abs(pred_rank - true_rank))
    print("Mean Index Error for " + str(len(Y_test)) + " test examples: ", meanDiff)
    print("Percent off: ", float(meanDiff) / len(Y_test) * 100)
    
    np.save(outputPath + "/pred_test/pred_" + train + "_" + test + str(epochs), prediction)
    np.save(outputPath + "/pred_test/test_" + train + "_" + test + str(epochs), Y_test)

    result = np.sqrt(mean_squared_error(Y_test, prediction))
    print("Testing RMSE: {}".format(result))


    print("Saving model to: ", weightsName) 
    reg.model.save(weightsName)
    return p
    
#Uganda: UGA Nigeria: NGA Tanzania: TZA Malawi: MWI Ghana: GHA
#if __name__ == "__main__":
def main():
    
    data = np.load("/atlas/u/esheehan/wikipedia_project/dataset/poverty_dataset/data/AfricaArticleClustersUpdated.npy")
    global num_articles
    #countries = ["UGA", "NGA", "TZA", "MWI", "GHA", "EGY", "africa"]
    countries = ["TZA", "GHA", "EGY", "africa"]
    for i in countries:
        train = i
        test = "TZA"
        epochs = 5
        if train == "africa":
            epochs = 10
        concat = True
        clean_split = False
        num = 1

        rootDir = "/atlas/u/esheehan/wikipedia_project/dataset/poverty_dataset/"

        outputPath = rootDir + "output/article_number_iterative_results/"
        weightsPath = rootDir + "output/article_number_iterative_results/weights/"
        logPath = rootDir + "output/article_number_iterative_results/logs/"
    
        filename = "Train" + train + "Test" + test + str(epochs)

        if not os.path.exists(outputPath): 
            print("{} does not exist, creating it now".format(outputPath))
            os.makedirs(outputPath)
            os.makedirs(weightsPath)
            os.makedirs(logPath)
            os.makedirs(outputPath + "/pred_test")
            os.makedirs(outputPath + "/results")

        # Set up logging 
        sys.stdout = open(logPath + filename + ".txt", 'w')
        print("Running model") 

        results = []

        while num_articles <= 10:
            X_train, X_test, Y_train, Y_test = load_data(data, train, test, concat, clean_split, num_articles)
            s = run_model(train, test, epochs, concat, clean_split, outputPath, \
                    weightsPath + filename + ".hdf5", num_articles, X_train, X_test, Y_train, Y_test)

            num_articles += 1
            results.append(s)

        np.save(outputPath + "/results/" + train + "_" + test, np.asarray(results))

        num_articles = 1

main()
