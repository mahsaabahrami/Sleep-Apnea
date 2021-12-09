#------------------------------------------------------------------------------------------------
# OBSTRUCTIVE SLEEP APNEA DETECTION FROM SINGLE-LEAD ECG: COMPREHENSIVE ANALYSIS OF DEEP LEARNING
                                            # WRITTEN BY: M. BAHRAMI
                                            # DATE: 2021
                                  # K. N. TOOSI UNIVERSTIY OF TECHNOLOGY
# ------------------------------------------------------------------------------------------------
# IMPORT LIBRARIES:
import pickle
import keras
import numpy as np
import os
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,Permute,Reshape,GRU,BatchNormalization,LSTM,Bidirectional
from keras.regularizers import l2
from scipy.interpolate import splev, splrep
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold,StratifiedShuffleSplit   
# ------------------------------------------------------------------------------------------------
class deep_models:
   def AlexNet(weight=1e-3):

    model= Sequential()
    model.add(Conv2D(96, kernel_size=(11,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,1)))   
    model.add(Conv2D(384, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Conv2D(384, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1))) 

    model.add(Flatten())
    model.add(Dense(209, activation="relu"))
    model.add(Dense(34, activation="relu"))

    model.add(Dense(2, activation="softmax"))
    return model

    def AlexNet_GRU(weight=1e-3):

     model= Sequential()
     model.add(Conv2D(96, kernel_size=(11,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1)))
     model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1)))   
     model.add(Conv2D(384, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(Conv2D(384, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1))) 

     model.add(Permute((2,1,3)))
     model.add(Reshape((1,5*256))) 

     model.add(GRU(183, return_sequences=True))
     model.add(Flatten())
     model.add(Dense(30, activation="relu"))

     model.add(Dense(2, activation="softmax"))
     return model
    def AlexNet_LSTM(weight=1e-3):

     model= Sequential()
     model.add(Conv2D(96, kernel_size=(11,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1)))
     model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1)))   
     model.add(Conv2D(384, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(Conv2D(384, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1))) 

     model.add(Permute((2,1,3)))
     model.add(Reshape((1,5*256))) 

     model.add(LSTM(183, return_sequences=True))
     model.add(Flatten())
     model.add(Dense(30, activation="relu"))

     model.add(Dense(2, activation="softmax"))
     return model
 
    def AlexNet_BiLSTM(weight=1e-3):
     
     model= Sequential()
     
     model.add(Conv2D(96, kernel_size=(11,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1)))
     model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1)))   
     model.add(Conv2D(384, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(Conv2D(384, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1))) 

     model.add(Permute((2,1,3)))
     model.add(Reshape((1,5*256))) 

     model.add(Bidirectional(LSTM(183, return_sequences=True)))
     
     model.add(Flatten())
     model.add(Dense(30, activation="relu"))
     model.add(Dense(2, activation="softmax"))
     return model
 
    def VGG16(weight=1e-3):

     model= Sequential()
     
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
   
     model.add(Flatten())
     model.add(Dense(512, activation="relu"))
     model.add(Dense(2, activation="softmax"))
     return model
 
    def VGG16_GRU(weight=1e-3):

     model= Sequential()

     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))

     model.add(Permute((2,1,3)))
     model.add(Reshape((1,7*512))) 

     model.add(GRU(512, return_sequences=True))
    
     model.add(Flatten())
     model.add(Dense(84, activation="relu"))

     model.add(Dense(2, activation="softmax"))
     return model
 
    def VGG16_LSTM(weight=1e-3):

     model= Sequential()

     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))

     model.add(Permute((2,1,3)))
     model.add(Reshape((1,7*512))) 

     model.add(LSTM(512, return_sequences=True))
    
     model.add(Flatten())
     model.add(Dense(84, activation="relu"))

     model.add(Dense(2, activation="softmax"))
     return model
    def VGG16_BiLSTM(weight=1e-3):

     model= Sequential()
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))

     model.add(Permute((2,1,3)))
     model.add(Reshape((1,7*512))) 

     model.add(Bidirectional(LSTM(512, return_sequences=True)))
    
     model.add(Flatten())
     model.add(Dense(84, activation="relu"))

     model.add(Dense(2, activation="softmax"))
     return model
 
    
    def VGG19(weight=1e-3):

     model= Sequential()
     
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     
     model.add(Flatten())
     
     model.add(Dense(167, activation="relu"))
     model.add(Dense(2, activation="softmax"))
     return model
 
    def VGG19_GRU(weight=1e-3):
    
     model= Sequential()
     
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Permute((2,1,3)))
     
     model.add(Reshape((1,2*512))) 

     model.add(GRU(146, return_sequences=True)) 
     
     model.add(Flatten())
     
     model.add(Dense(24, activation="relu"))
     model.add(Dense(2, activation="softmax"))
     return model
 
    def VGG19_LSTM(weight=1e-3):

     model= Sequential()
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Permute((2,1,3)))
     model.add(Reshape((1,2*512))) 

     model.add(LSTM(146, return_sequences=True)) 
     
     model.add(Flatten())
     model.add(Dense(24, activation="relu"))
     model.add(Dense(2, activation="softmax"))
     return model
 
    def VGG19_BiLSTM(weight=1e-3):

     model= Sequential()
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
     model.add(Conv2D(64, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(128, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(256, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     model.add(MaxPooling2D(pool_size=(2,1)))
     model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
     
     
     model.add(Permute((2,1,3)))
     model.add(Reshape((1,2*512))) 

    model.add(Bidirectional(LSTM(146, return_sequences=True)))
    
    model.add(Flatten())
    model.add(Dense(24, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model

   def ZFNet(weight=1e-3):

    model= Sequential()
    model.add(Conv2D(96, kernel_size=(7,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
    model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1))) 
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(1024, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1)))  
    
    model.add(Flatten())
    
    model.add(Dense(418, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model

   def ZFNet_GRU(weight=1e-3):

    model= Sequential()
    model.add(Conv2D(96, kernel_size=(7,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
    model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1))) 
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(1024, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1)))
    model.add(Permute((2,1,3)))
    model.add(Reshape((1,5*512))) 

    model.add(GRU(365, return_sequences=True)) 
    model.add(Flatten())
    model.add(Dense(60, activation="relu"))

    model.add(Dense(2, activation="softmax"))
    return model

   def ZFNet_LSTM(weight=1e-3):

    model= Sequential()
    model.add(Conv2D(96, kernel_size=(7,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
    model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1))) 
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(1024, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1)))
    model.add(Permute((2,1,3)))
    model.add(Reshape((1,5*512))) 

    model.add(LSTM(365, return_sequences=True)) 
    model.add(Flatten())
    model.add(Dense(60, activation="relu"))

    model.add(Dense(2, activation="softmax"))
    return model
   def ZFNet_BiLSTM(weight=1e-3):

    model= Sequential()
    model.add(Conv2D(96, kernel_size=(7,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,1,2)))
    model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1))) 
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3,1), strides=(1,1), padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(1024, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Conv2D(512, kernel_size=(3,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1),strides=(2,1)))
    model.add(Permute((2,1,3)))
    model.add(Reshape((1,5*512))) 

    model.add(Bidirectional(LSTM(365, return_sequences=True))) 
    model.add(Flatten())
    model.add(Dense(60, activation="relu"))

    model.add(Dense(2, activation="softmax"))
    return model
