##----------------------------------------------------------------------------      
## SLEEP APNEA DETECTION: COMPREHENSIVE ANALYSIS OF MACHINE LEARNING AND DEEP LEARNING METHODS
                                    ## WRITTEN BY: M.BAHRAMI
                                        ## DATE: 12-6-2021
                                ## MODEL: HYBRID MODEL OF ZFNet-LSTM
##-----------------------------------------------------------------------------
#IMPORT LIBRARIES    
import pickle
import numpy as np
import os
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,BatchNormalization,LSTM
from keras.regularizers import l2
from scipy.interpolate import splev, splrep
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# DEEP LEARNING MODELS NEED TO VECTORS OR MATRIX WITH SAME SIZE
# R-R INTERVALS DONT HAVE SAME SIZE, SO WE NEED TO INTERPOLATE VECTORS TO GET VECTORS WITH SAME SIZE.
# BASED ON OUR EXPERIENCE INTERPOLATION IN 3 HZ BETTER AND ACCURATE.
ir = 3 # INTERPOLATION RATE(3HZ)
time_range= 60 # 60-s INTERVALS OF ECG SIGNALS
weight=1e-3 #  WEIGHT L2 FOR REGULARIZATION(AVODING OVERFITTING PARAMETER)
#------------------------------------------------------------------------------
# NORMALIZATION:
# DEEP LEARNING AND EVEN NEURAL NETWORKS INPUT SHOULD BE NORMALIZED:
# MIN-MAX METHOD APPLIED FOR SCALING:(Array-min(Array))/(max(Array)-min(Array))
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#------------------------------------------------------------------------------
# FIRSTLY WE PRE-PROCESSED OUR DATA IN "apnea-detection_TD3_Kfold.pkl" FILE
# IN PRE-PROCESSING SECTION WE EXTRACT R-R INTERVALS AND R-PEAK AMPLITUDES 
# IN THIS PART WE LOAD THIS DATA AND INTERPOLATE AND CONCATE FOR FEEDING TO NETWORKS
def load_data():    
    tm = np.arange(0, (time_range), step=(1) / float(ir)) # TIME METRIC FOR INTERPOLATION 
    # LOAD AND INTERPOLATE R-R INTERVALS AND R-PEAK AMPLITUDES
    with open(os.path.join( "apnea-detection_TD3_Kfold.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)

    x = []
    X, Y = apnea_ecg["X"], apnea_ecg["Y"]
    for i in range(len(X)):
        (rri_tm, rri_signal), (amp_tm, amp_signal) = X[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1) 
        amp_interp_signal = splev(tm, splrep(amp_tm, scaler(amp_signal), k=3), ext=1)
        x.append([rri_interp_signal, amp_interp_signal])
    x = np.array(x, dtype="float32")
    
    
    x = np.expand_dims(x,1)
    x_final=np.array(x, dtype="float32").transpose((0,3,1,2))
    
    
    return x_final, Y
#------------------------------------------------------------------------------
# CREAT DEEP LEARNING MODEL
def create_model(weight=1e-3):

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

#------------------------------------------------------------------------------
# Define learning rate schedule for preventing overfitting in deep learning methods:
def lr_schedule(epoch, lr):
    if epoch > 30 and \
            (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Compile and evaluate model: 
if __name__ == "__main__":
    # loading Data:
    X, Y = load_data()
    # we have labels(Y) in a binary way 0 for normal and 1 for apnea patients
    # we want to classify data into 2-class so we changed y in a categorical way:
    Y = tf.keras.utils.to_categorical(Y, num_classes=2)
    # we used k-fold cross-validation for more reliable experiments: 
    kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=7)
    cvscores = []
    ACC=[]
    SN=[]
    SP=[]
    F2=[]
    # separate train& test and then compile model
    for train, test in kfold.split(X, Y.argmax(1)):
     model = create_model()

     
     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
     # define callback for early stopping:
     lr_scheduler = LearningRateScheduler(lr_schedule)
     callback1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
     
     #10% of Data used for validation:
     X1,x_val,Y1,y_val=train_test_split(X[train],Y[train],test_size=0.10)
     
     history = model.fit(X1, Y1, batch_size=128, epochs=100, validation_data=(x_val, y_val),
                        callbacks=[callback1,lr_scheduler])
    
     model.save(os.path.join("model.KF_AlexNet_LSTM.h5"))
     
     loss, accuracy = model.evaluate(X[test], Y[test]) 

     y_score = model.predict(X[test])
     
     y_predict= np.argmax(y_score, axis=-1)
     y_training = np.argmax(Y[test], axis=-1)
     # Confusion matrix:
     from sklearn.metrics import confusion_matrix
     from sklearn.metrics import f1_score
     C = confusion_matrix(y_training, y_predict, labels=(1, 0))
     TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
     acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
     f2=f1_score(y_training, y_predict)
     
     ACC.append(acc * 100)
     SN.append(sn * 100)
     SP.append(sp * 100)
     F2.append(f2 * 100)
