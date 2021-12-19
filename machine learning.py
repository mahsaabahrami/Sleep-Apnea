#------------------------------------------------------------------------------
# OBSTRUCTIVE SLEEP APNEA DETECTION FROM SINGLE-LEAD ECG: COMPREHENSIVE ANALYSIS OF MACHINE LEARNING
                                           
                                            # DATE: 2021
                                    # Machine learning models   
# -----------------------------------------------------------------------------
# IMPORT LIBRARIES:
# WE USED SCIKIT-LEARN PACKAGE FOR IMPLIMENTING MACHINE LEARNING METHODS:
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
#------------------------------------------------------------------------------
# LOAD DATASET:
    # WE PRE-PROCESSED OUR DATA AND EXTRACTED 30 FEATURES FROM EACH SEGMENTS OF ECG SIGNAL
    
feature_set = pd.read_csv(r'''D:\DATA_set.csv''')
dataset=feature_set.values
X = dataset[:,:30].astype('float32') # FEATURES
Y= dataset[:,30].astype('float32')   # LABLES
#------------------------------------------------------------------------------
# CLF: CLASSIFIERS
# WE IMPLIMENTED 14 MACHINE LEARNING METHODS

clf1 = LinearDiscriminantAnalysis(solver='svd')
clf2 = QuadraticDiscriminantAnalysis(tol=8.30E-4)
clf3 = svm.SVC(kernel='rbf', C=252, gamma=0.75)
clf4 = KNeighborsClassifier(n_neighbors=11, metric='manhattan',leaf_size=70)
clf5= RandomForestClassifier(n_estimators=90, max_depth=11)
clf6= DecisionTreeClassifier(max_depth=5)
clf7= ExtraTreesClassifier(n_estimators=80, max_depth=28)
clf8= MLPClassifier(hidden_layer_sizes=(50),activation='relu',max_iter=500)
clf9 = GradientBoostingClassifier(n_estimators=80, max_depth=24,subsample = 0.59, criterion = 'mse',loss= 'exponential')
clf10 = LogisticRegression(C=95.78)
clf11 = AdaBoostClassifier(n_estimators=70,learning_rate=0.39)
clf12 = GaussianNB(var_smoothing=0.00946)
clf13 = GaussianProcessClassifier(max_iter_predict=3,warm_start= True)
clf = VotingClassifier(
   estimators=[('C1',clf1),('C2',clf2),('C3',clf3),('C4',clf4),('C5',clf5),('C6',clf6),
               ('C7',clf7),('C8',clf8),('C9',clf9),('C10',clf10),('C11',clf11),('C12',clf12),('C13',clf13)],
 voting='hard')
#------------------------------------------------------------------------------
# OUR DATASET IS IMBALANCED SO, WE USED STRATIFIED K-FOLD FOR BALANCING IT
# WE USED 5-FOLD CROSS-VALIDATION METHOD
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
#------------------------------------------------------------------------------
# WE DEFIEND ACC,SN,SP, AND F2 FOR SAVING THEM IN ALL 5-FOLDS
ACC=[]
SN=[]
SP=[]
F2=[]
#------------------------------------------------------------------------------
# TRAIN, TEST SPLIT: 
for train, test in kfold.split(X, Y):

     # TRAINING: PRE-PROCESSING CONSIST OF MIN-MAX NORMALIZATION, PCA DIMENSION REDUCTION   
     scaler = preprocessing.MinMaxScaler().fit(X[train])
     X_train_transformed = scaler.transform(X[train])
     pc= PCA(n_components=0.98,svd_solver = 'full')
     pca=pc.fit(X_train_transformed)
     X_pCA=pca.transform(X_train_transformed)
     clf.fit(X_pCA, Y[train])
     # TESTING:
     X_test_transformed = scaler.transform(X[test])
     X_test_PCA=pca.transform(X_test_transformed)
     y_score = clf.predict(X_test_PCA)
     # COMPUTE CONFUSION MATRIX: 
     from sklearn.metrics import f1_score
     C = confusion_matrix(Y[test], y_score, labels=(1, 0))
     TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
     acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
     f2=f1_score(Y[test], y_score)
     # APPENDING ACC,SN, SP, AND F2
     ACC.append(acc * 100)
     SN.append(sn * 100)
     SP.append(sp * 100)
     F2.append(f2 * 100)
#------------------------------------------------------------------------------
