import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Concatenate, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import sys
import numpy as np
from scipy.misc import imshow, imresize
from PIL import Image, ImageEnhance

data = np.load("/atlas/u/esheehan/wikipedia_project/dataset/GUF_dataset/data/AfricaArticleClustersUpdated.npy")
titleMap = pd.read_csv('/atlas/u/esheehan/wikipedia_project/CNN/Africa_Image_Coordinates.csv').dropna()
imageRoot = "/atlas/u/esheehan/wikipedia_project/dataset/image_dataset/Africa_Images/"

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def load_data():
    # Title ex: 0,""Pedro Arrupe" Political Training Institute",institute,38.13750555555556,13.333297222222226
    # Image format location: imageRoot + category + id
    np.random.seed(1234)

    print("Num examples: ", len(data))
    noImage = 0 
    
    X,Y = [], []
    for i in data:
        if i[1] == "NGA":
            closest3All = i[6][:3]
            closest3Filename = []
            for d in closest3All:
                f = titleMap[titleMap['name'] == d[1]]
                if f.empty:
#                     print("No file found for: ", d[1], file=sys.stderr)
                    f = titleMap[titleMap['name'] == d[1] + "_0"]
                    if f.empty:
                        noImage += 1 
                        continue 
                categoryName = f["category"].iloc[0]
                categoryName.replace(" ", "\ ")
                curFilename = imageRoot + categoryName + "/" +  str(f["id"].iloc[0]) + ".jpeg"
                # print(curFilename)
                closest3Filename.append(curFilename)   
                # Currently returns filenames, should load them as images??
            if len(closest3Filename) == 3:
                for augmentCnt in range(5):
                    curImages = []
                    for file in closest3Filename:
                        im = Image.open(file).convert('RGB')
                        im = im.resize((224,224))
                        im = im.rotate(np.random.randint(0,360))
                        im = ImageEnhance.Brightness(im).enhance(np.random.uniform())
                        im = ImageEnhance.Sharpness(im).enhance(np.random.uniform(0.0, 2.0))
                        im2arr = np.array(im) # im2arr.shape: height x width x channel
                        curImages.append(im2arr)
                    X.append(curImages)
                    Y.append(i[2])
            

    print("No images found for {} examples.".format(noImage))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1234, shuffle=True)
    X_train, X_test, Y_train, Y_test = np.asarray(X_train), np.asarray(X_test), np.asarray(Y_train), np.asarray(Y_test)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)
    print("Y_train Shape: ", Y_train.shape)
    print("Y_test Shape: ", Y_test.shape)

    print("Data Loaded") 
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_data()
print("Dataset Loaded")

def load_model():
    
    np.random.seed(1234)
    
    bigInput = Input(shape=(3,224,224,3), dtype='float32', name='bigInput') 
    input1 = Lambda(lambda x: x[:,0])(bigInput)
    input2 = Lambda(lambda x: x[:,1])(bigInput)
    input3 = Lambda(lambda x: x[:,2])(bigInput)
    
    print(bigInput.shape, input1.shape, input2.shape, input3.shape)

    x = VGG16(weights='imagenet', include_top=False)
    x.name = 'vgg_1'
    x = x(input1)
    a = VGG16(weights='imagenet', include_top=False)
    a.name = 'vgg_2'
    a = a(input2)
    b = VGG16(weights='imagenet', include_top=False)
    b.name = 'vgg_3'
    b = b(input3)

    # INPUT 1
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid')(x)
    x = Flatten()(x)

    # INPUT 2
    a = MaxPooling2D(pool_size=(2, 2))(a)
    a = Conv2D(3, (3, 3), activation='sigmoid')(a)
    a = Flatten()(a)
    
    # INPUT 3
    b = MaxPooling2D(pool_size=(2, 2))(b)
    b = Conv2D(3, (3, 3), activation='sigmoid')(b)
    b = Flatten()(b)
    
    c = keras.layers.concatenate([x, a, b])
    c = Dense(64)(c)
    main_output = Dense(1, name='main_output')(c)
    print(main_output.shape)
    
    model = Model(inputs=[bigInput], outputs=[main_output])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination,'mae'])
    print(model.summary())
    return model

def run_model():

    reg = load_model()
    print("Model Loaded")
    reg= KerasRegressor(build_fn=load_model, epochs=10, verbose=1,validation_split=0.2)

    print("Testing model")
    reg.fit(X_train, Y_train)
    prediction=reg.predict(X_test)
    print("R2: ", r2_score(Y_test, prediction))
    print("Pearson's r: ", pearsonr(Y_test, prediction))
    result = np.sqrt(mean_squared_error(Y_test, prediction))
    print("Testing RMSE: {}".format(result))


# In[32]:



if __name__ == "__main__":

    run_model()


