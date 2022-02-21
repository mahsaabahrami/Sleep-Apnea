# Sleep Apnea Detection from Single-Lead ECG: A Comprehensive Analysis of Machine Learning and Deep Learning Algorithms
In this project, we implemented a number of machine learning and deep learning methods for sleep apnea detection.
Conventional machine learning methods have four main steps for sleep apnea detection: pre-processing, feature extraction, feature selection, and classification.
After extracting R-peaks from ECG signal, we extracted time, frequency, and non-linear features of ECG signal. Principal component analysis then applied for dimension reduction.
Linear discriminate analysis (LDA), Quadratic discriminate analysis (QDA), Support vector machine (SVM), Naive Bayes (NB), Multi-layer perceptron (MLP),Decision tree (DT), Extra tree (ET), Random forest (RF), Adaboost, Gradient boosting (GB), Logistic regression (LR), and Majority voting (MV) methods are well-known conventional machine learning methods that were  implemented and compared for sleep apnea detection from single-lead ECG.
And finally, differnet well-known deep learning methods were modified and used for sleep apnea detection. In this regard, we modified AlexNet, ZFNet, VGG16, and VGG19 for sleep apnea detection. We also, applied deep recurrent neural networks consist of LSTM, GRU, and BiLSTM and hybrid models of CNN and deep recurrent neural networks.


Here, we used PhysioNet ECG Database which is available in: https://physionet.org/content/apnea-ecg/1.0.0/ 
Firstly, we extracted R-R Intervals and R-peak amplitude and then fed to machine learning and deep learning methods.




# Papers:

If this work is helpful in your research, please consider starring ⭐ us and citing our papers which provide a comprehensive analysis of deep learning and machine learning methods for sleep apnea detection:

1- Bahrami, Mahsa, and Mohamad Forouzanfar. "Detection of sleep apnea from single-lead ECG: Comparison of deep learning algorithms." In 2021 IEEE International Symposium on Medical Measurements and Applications (MeMeA), pp. 1-5. IEEE, 2021.

This paper can be downloded at: https://ieeexplore.ieee.org/abstract/document/9478745

2- Bahrami, Mahsa, and Mohamad Forouzanfar. "Sleep Apnea Detection from Single-Lead ECG: A Comprehensive Analysis of Machine Learning and Deep Learning Algorithms." IEEE Transactions on Instrumentation and Measurement (2022).

This paper can be downloded at: https://ieeexplore.ieee.org/abstract/document/9714370





# Requirements:

1-scikit-learn

2-hrvanalysis

3-numpy

4-keras

5-tensorflow

6-scipy


# References:

1- hrvanalysis package for feature extration: https://github.com/Aura-healthcare/hrv-analysis

2- scikit-learn package for machine learning: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org/stable/

3- keras for deep learning: https://keras.io/  





