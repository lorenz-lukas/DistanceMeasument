#!/usr/bin/python
# coding=utf-8
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16 as vgg
from keras.applications.vgg16 import preprocess_input
import os, sys
import cv2, numpy as np
from random import shuffle
import pandas as pd

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
    #cv2.namedWindow('UNDISTORT')  # Create a named window
    #cv2.moveWindow('UNDISTORT', 20, 20)
    while img!=[]:
        image = img[-1]
        while image!=[]:
            im = cv2.imread('test1/out-test1_{}/{}'.format(j,str(image[-1])))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (940,540))#.astype(np.float32)#(224, 224)
            im = im[y:y+224, x:x+224] #CROP
            #print im.shape
            #cv2.imshow('UNDISTORT', im)
            #cv2.waitKey(3)
            #im[:,:,0] -= 103.939
            #im[:,:,1] -= 116.779
            #im[:,:,2] -= 123.68
            #im = im.transpose((2,0,1))
            #im = np.expand_dims(im, axis=0)
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
    #le = LabelEncoder().fit(d)
    #d = np_utils.to_categorical(le.transform(d), 344)
    return d

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224, 224, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu')) #linear
    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer='normal', activation='relu')) #linear
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='relu')) #linear

    if weights_path:
        model.load_weights(weights_path,by_name=True)

    return model

def fine():
    model = vgg(weights='imagenet', include_top=False)
    x = model.output
    x = Flatten(input_shape = (512,3,3))(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, kernel_initializer='normal', activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, kernel_initializer='normal', activation='relu')(x)
    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    conv_base.trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def train(model, testX, trainX, testY, trainY,):

    model.compile(optimizer='adam',loss='mean_squared_error')
    number_epochs = 1
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=1, epochs=number_epochs, verbose=1)
    checkpoint = ModelCheckpoint("./weights/VGG16.h5", monitor="val_loss", verbose=1)
    #callbacks = [checkpoint]

    return H, model

def evaluate_model(model, testY, testX, le):
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=le.classes_))

def plot_val_acc(H, n_epochs):
    n_epochs = 5
    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["acc"], label="acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy for VGG16")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
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

    #print len(img)
    #print len(label)
    (trainX, testX, trainY, testY) = train_test_split(img, label, test_size=0.2, random_state=42)
    trainX = trainX[0:int(len(trainX)/100)]
    trainY = trainY[0:int(len(trainY)/100)]
    testX = testX[0:int(len(testX)/100)]
    testY = testY[0:int(len(testY)/100)]
    print 'Test pretrained model:\n\n\n'
    model = VGG_16('vgg16_weights.h5')
    #model = fine()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    print 'Training model:\n\n\n'
    H, model = train(model, testX, trainX, testY, trainY)
    evaluate_model(model,testX,testY,label)

    #(loss='mean_absolute_error', optimizer='rmsprop')
    #(optimizer=sgd, loss='categorical_crossentropy')
    #out = model.predict(im)
    #print np.argmax(out)
