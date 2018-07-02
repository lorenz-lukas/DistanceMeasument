#!/usr/bin/python
# coding=utf-8
import os, sys
import cv2, numpy as np
from random import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def data():
    img = []
    label = []
    temp = []
    for i in xrange(1,4):
        im_list = os.listdir('test1/out-test1_{}'.format(i))
        img.append(im_list)

    for i in xrange(1,4):
        data= pd.read_csv('test1/txt/test1_{}_1.txt'.format(i), sep=";", header=None)
        label.append(data[1:len(data)])
    while label != []:
        temp.append(label[-1])
        del label[-1]
    del im_list
    del data
    return temp, img

def load_img(img):
    i = []
    j = 3
    y = 130
    x = 340
    while img!=[]:
        image = img[-1]
        while image!=[]:
            im = cv2.imread('test1/out-test1_{}/{}'.format(j,str(image[-1])))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (940,540))#.astype(np.float32)#(224, 224)
            im = im[y:y+224, x:x+224] #CROP
            im = img_to_array(im)
            i.append(im)
            del image[-1]
            del im
        del img[-1]
        j= j-1
    im = np.array(i, dtype="float") / 255.0
    return im

def load_label(data):
    d = []
    while (data!=[]):
        td = data[-1] # Time and Distance
        #t = td[0] # time in ms
        dis = td[1]#np.array(td[1], dtype="float")
        for i in xrange(1,len(dis)):
            d.append(dis[i])
        del data[-1]
    del td
    d = np.array(d)
    return d

def reg_model():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224, 224, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main():
    print '####################################################'
    print '##                  Opencv3.4.0                   ##'
    print '##                   python2.7                    ##'
    print '##                     VGG16                      ##'
    print '####################################################'
    print '\n\n\n'

    print 'Loading data ...\n'
    label, img = data()
    img = load_img(img)
    label = load_label(label)

    (trainX, testX, trainY, testY) = train_test_split(img, label, test_size=0.2, random_state=42)
    print 'Test pretrained model:\n\n\n'
    model = reg_model()
    print 'Training model:\n\n\n'
    # fix random seed for reproducibility
    seed = 7
    # evaluate model with standardized dataset
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=reg_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    print 'Crossvalidation:\n\n\n'
    # encode class values as integers
    results = cross_val_score(pipeline, img, label, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

main()
