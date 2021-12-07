#------------------------------------------------------------------------------
# OBSTRUCTIVE SLEEP APNEA DETECTION FROM SINGLE-LEAD ECG: COMPREHENSIVE ANALYSIS OF MACHINE LEARNING
                                            # WRITTEN BY: M. BAHRAMI
                                            # DATE: 2021
                      # K. N. TOOSI UNIVERSTIY OF TECHNOLOGY AND école de technologie supérieure
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
#------------------------------------------------------------------------------
# LOAD DATASET:
    # WE PRE-PROCESSED OUR DATA AND EXTRACTED 30 FEATURES FROM EACH SEGMENTS OF ECG SIGNAL
    
feature_set = pd.read_csv(r'''D:\DATA_set.csv''')
dataset=feature_set.values
X = dataset[:,:30].astype('float32') # FEATURES
Y= dataset[:,30].astype('float32')   # LABLES
#------------------------------------------------------------------------------
# CLF: CLASSIFIERS
# WE IMPLIMENTED 13 MACHINE LEARNING METHODS
# HYPER-PARAMETERS OF MACHINE LEARNING METHODS WERE COMPUTED WITH TRIAL AND ERROR

clf1 = LinearDiscriminantAnalysis()
clf2 = QuadraticDiscriminantAnalysis()
clf3 = svm.SVC(kernel='linear', C=1)
clf4 = KNeighborsClassifier(n_neighbors=15)
clf5= RandomForestClassifier(max_depth=10, random_state=0)
clf6= DecisionTreeClassifier(max_depth=5, min_samples_split=5,random_state=1)
clf7= ExtraTreesClassifier(n_estimators=20, max_depth=15, min_samples_split=2, random_state=1)
clf8= MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 2), random_state=0)
clf9 =  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf10 =  LogisticRegression(multi_class='multinomial', random_state=1)
clf11 = AdaBoostClassifier()
clf12 = GaussianNB()
clf = VotingClassifier(
   estimators=[('C1',clf1),('C2',clf2),('C3',clf3),('C1',clf4),('C2',clf5),('C3',clf6),
               ('C1',clf7),('C2',clf8),('C3',clf9),('C1',clf10),('C2',clf11),('C3',clf12)],
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

     # TRAINING: PRE-PROCESSING CONSIST OF MIN-MAX NORMALIZATION, PCA DIMENSION REDUCTIO   
     scaler = preprocessing.MinMaxScaler().fit(X[train])
     X_train_transformed = scaler.transform(X[train])
     pc= PCA(n_components=0.98,svd_solver = 'full')
     pca=pc.fit(X_train_transformed)
     X_pCA=pca.transform(X_train_transformed)
     clf2.fit(X_pCA, Y[train])
     # TESTING:
     X_test_transformed = scaler.transform(X[test])
     X_test_PCA=pca.transform(X_test_transformed)
     y_score = clf2.predict(X_test_PCA)
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