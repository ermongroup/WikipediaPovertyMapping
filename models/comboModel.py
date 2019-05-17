import os
import sys

import keras
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D, Input, Concatenate,concatenate
from keras.layers import Conv2D, AveragePooling2D, Reshape
from keras import regularizers

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import math

from keras.applications.densenet import DenseNet121

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

import tensorflow as tf
#from keras.utils.training_utils import multi_gpu_model

from keras import backend as K
print(str(K.tensorflow_backend._get_available_gpus()))

output_path = "/atlas/u/esheehan/wikipedia_project/dataset/poverty_dataset/output/CNN_Final_Out/"
doc2vec_path = '/atlas/u/esheehan/wikipedia_project/dataset/poverty_dataset/data/Full_Clusters_Updated.npy'
NUM_EPOCHS = 15

def createPlot(x, y, label, xName, yName, tag):
    plt.figure()
    plt.plot(x, y, '-o')
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.savefig(output_path + tag + "/plots/" + label.replace(" ", "_"))

def visualizeHistory(history, tag):

    print(history.keys())

    createPlot(range(NUM_EPOCHS), history['loss'], "Train Loss", "Epoch", "Loss", tag)
    createPlot(range(NUM_EPOCHS), history['r2'], "Train R2", "Epoch", "R2", tag)
    createPlot(range(NUM_EPOCHS), history['pearson_r2'], "Train Pearson's r2", "Epoch", "Pearson's r2", tag)

    createPlot(range(NUM_EPOCHS), history['val_loss'], "Test Loss", "Epoch", "Loss", tag)
    createPlot(range(NUM_EPOCHS), history['val_r2'], "Test R2", "Epoch", "R2", tag)
    createPlot(range(NUM_EPOCHS), history['val_pearson_r2'], "Test Pearson's r2", "Epoch", "Pearson's r2", tag)

def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def pearson_r2(y_true, y_pred):
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

    result=top/(bottom + K.epsilon())
    result = K.mean(result)
    return K.pow(result, 2)

def concatenate_doc2vec(i, num=10):
    embeddings = []
    for index, article in enumerate(i[6]):
        embeddings += list(article[4]) + [article[0]]
        if index == num - 1:
            break
    return embeddings

def load_model_cnn2():

    print("Loading model...")

    np.random.seed(1234)
    nightlight = Input(shape=(255,255,1), name='nightlight')
    y = Conv2D(3, 5)(nightlight)
    y = Conv2D(6, 5)(y)
    y = Flatten()(y)
    y = Dense(512, kernel_initializer='normal', activation='relu')(y)
    y = Dense(256, kernel_initializer='normal', activation='relu')(y)

    doc2vec = Input(shape=(3010,), name = "doc2vec")
    z = Dense(512, kernel_initializer='normal', activation='relu', input_shape=(3010,))(doc2vec)
    z = Dense(512, kernel_initializer='normal', activation='relu')(z)
    z = Dense(256, kernel_initializer='normal', activation='relu')(z)

    merge = concatenate([y, z])

    merge = Dense(512, kernel_initializer='normal', activation='relu')(merge)
    merge = Dense(256, kernel_initializer='normal', activation='relu')(merge)
    merge = Dense(128, kernel_initializer='normal', activation='relu')(merge)
    merge = Dense(32, kernel_initializer='normal', activation='relu')(merge)
    merge = Dense(1, kernel_initializer='normal',  name='merge')(merge)
     
    model = Model(inputs=[nightlight, doc2vec], outputs=[merge])
    #adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer="adam", metrics=[r2, pearson_r2])
    print(model.summary())
    return model

def get_image_dataset(data):

    rgbn = np.asarray([np.load("/atlas/u/esheehan/wikipedia_project/dataset/poverty_dataset/data/" + i[7] + ".npy") for i in data])
    print(str(rgbn.shape))
    np.nan_to_num(rgbn)
    nightlights = rgbn[:,3,:,:]

    #nightlights = np.asarray([np.stack([i, i, i], axis=-1) for i in nightlights])
    nightlights = np.reshape(nightlights, (-1, 255, 255, 1))
    print(str(nightlights.shape))
    rgb = rgbn[:,:3,:,:]
    rgb = np.transpose(rgb, (0, 2, 3, 1))
    print(str(rgb.shape))

    return rgb, nightlights

def load_data(train, test):

    print("Loading data...")
    X_train_rgb, X_train_nightlights = get_image_dataset(train)
    X_train_embeddings = np.asarray([concatenate_doc2vec(i) for i in train])
    #X_train = [X_train_rgb, X_train_nightlights, X_train_embeddings]
    X_train = [X_train_nightlights, X_train_embeddings]

    Y_train = np.asarray([i[2][0] for i in train])

    X_test_rgb, X_test_nightlights = get_image_dataset(test)
    X_test_embeddings = np.asarray([concatenate_doc2vec(i) for i in test])
    #X_test = [X_test_rgb, X_test_nightlights, X_test_embeddings] 
    X_test = [X_test_nightlights, X_test_embeddings]

    Y_test = np.asarray([i[2][0] for i in test])

    return X_train, Y_train, X_test, Y_test

def train_model(train, test, tag, save=True):

    X_train, Y_train, X_test, Y_test = load_data(train, test)
    model = load_model_cnn2()
    if save:
        history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=NUM_EPOCHS, validation_data=(X_test, Y_test), shuffle=True, verbose=2)
        y = model.predict(X_train)
        print("Final Train R2: " + str(r2_score(Y_train, y)))
        print("Final Train Pearson r2: " + str(pearsonr(Y_train, y)[0]**2))
        print("Final Train Spearman rho: " + str(spearmanr(Y_train, y)[0]))

        y = model.predict(X_test)
        print("Final Test R2: " + str(r2_score(Y_test, y)))
        print("Final Test Pearson r2: " + str(pearsonr(Y_test, y)[0]**2))
        print("Final Test Spearman rho: " + str(spearmanr(Y_test, y)[0]))
        np.save(output_path + tag + "/y_true", Y_test)
        np.save(output_path + tag + "/y_pred", y)
    else:
        history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=NUM_EPOCHS, shuffle=True, verbose=2)

    return model, history

if __name__ == "__main__":

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    #sys.stdout = open(output_path + "log.txt", 'w')

    coordinates = np.load(doc2vec_path)

    #africa = ["ghana", "malawi", "nigeria", "tanzania", "uganda", "drc", "guinea", "kenya", "rwanda", "angola", "benin", \
            #"cameroon", "ethiopia", "guyana", "mali", "senegal", "zambia", "burkina", "cote", "lesotho", "mozambique", \
            #"sierra", "togo", "zimbabwe"]

    africa = ["ghana", "malawi", "nigeria", "tanzania", "uganda"]

    for country1 in africa:
        # Intra-country training/testing
        """ # Train and test on the same country
        tag = "Train_" + country1 + "_Test_" + country1
        print("Training on " + country1 + ", testing on " + country1)
        if not os.path.isdir(output_path + tag):
            os.mkdir(output_path + tag)
        recent = None
        for i in coordinates:
            if i[1] == country1:
                if recent is None or recent > i[-1]:
                    recent = i[-1]
        print(str(recent))
        data = [i for i in coordinates if i[1] == country1 and not (i[1] == "nigeria" and i[-1] == 2015)]
        shuffle(data)
        train, test = data[:int(len(data) * .8)], data[int(len(data) * .8):]

        model, __ = train_model(train, test, tag)
        del(train)
        del(test)
        del(model)"""

    for country1 in africa:
        # Training-testing across national boundaries exclusively
        print("Pretraining_" + country1)

        train = [i for i in coordinates if i[1] == country1]
        test = train
        model, __ = train_model(train, test, "", save=False)
        print("Done pretraining..")

        for country2 in africa:

            if country1 == country2:
                continue
            else:
                test = [i for i in coordinates if i[1] == country2]
            print("Training on " + country1 + ", testing on " + country2)
            tag = "Train_" + country1 + "_Test_" + country2

            ___, ___, X_test, Y_test = load_data(train, test)

            y = model.predict(X_test)
            print("Final Test R2: " + str(r2_score(Y_test, y)))
            print("Final Test Pearson r2: " + str(pearsonr(Y_test, y)[0]**2))
            print("Final Test Spearman rho: " + str(spearmanr(Y_test, y)[0]))
   
            if not os.path.isdir(output_path + tag):
                os.mkdir(output_path + tag)
            np.save(output_path + tag + "/y_true", Y_test)
            np.save(output_path + tag + "/y_pred", y)

        # Full train-test across all countries (intra and cross-boundary)
    """for country1 in africa:
        for country2 in africa:

            print("Training on " + country1 + ", testing on " + country2)
            tag = "Train_" + country1 + "_Test_" + country2
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            if country1 == country2:
                data = [i for i in coordinates if i[1] == country1]
                shuffle(data)
                train, test = data[:int(len(data) * .8)], data[int(len(data) * .8):]
            else:
                train = [i for i in coordinates if i[1] == country1]
                test = [i for i in coordinates if i[1] == country2]
            
            if not os.path.isdir(output_path + tag):
                os.mkdir(output_path + tag)
            model, history = train_model(train, test, tag) 

            if not os.path.isdir(output_path + tag + "/plots"):
                os.mkdir(output_path + tag + "/plots")
            visualizeHistory(history.history, tag)"""




