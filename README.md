# Obstructive Sleep Apnea Detection from Single-Lead ECG: A Comprehensive Analysis of Machine Learning and Deep Learning Algorithms
In this project, we implemented a number of machine learning and deep learning methods for sleep apnea detection.
Conventional machine learning methods have four main steps for sleep apnea detection: pre-processing, feature extraction, feature selection, and classification.
After extracting R-peaks from ECG signal, we extracted time, frequency, and non-linear features of ECG signal. Principal component analysis then applied for dimension reduction.
Linear discriminate analysis (LDA), Quadratic discriminate analysis (QDA), Support vector machine (SVM), Naive Bayes (NB), Multi-layer perceptrun (MLP),Decision tree (DT), Extra tree (ET), Random forest (RF), Adaboost, Gradient boosting (GB), Logistic regression (LR), and Majority voting (MV) methods are well-known conventional machine learning methods that were implimented and compared for sleep apnea detection from single-lead ECG.
And finally, differnet well-known deep learning methods were modified and used for sleep apnea detection. In this regard, we modified AlexNet, ZFNet, VGG16, and VGG19 for sleep apnea detection. We also, applied deep recurrent neural networks consist of LSTM, GRU, and BiLSTM and hybrid models of CNN and deep recurrnet neural networks.


Here, we used PhysioNet ECG Database which is available in: https://physionet.org/content/apnea-ecg/1.0.0/ 
Firstly, we extracted R-R Intervals and R-peak amplitude and then fed to machine learning and deep learning methods.
For running each deep learning models, open model and run it! and for machine learning, firstly, run feature extraction.py and then run machine learning.py!



You can also find and use all deep models models in Deep_MODELS.py class, which can be used in other applications.  
