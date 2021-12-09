import pickle
import os
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_frequency_domain_features
from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
from entropy import *
base_dir =  r'''D:\apnea-ecg-database-1.0.0'''
with open(os.path.join(base_dir, "Detection_ML_ECG_TIM.pkl"), 'rb') as f: # read preprocessing result
    apnea_ecg = pickle.load(f)
    x = []
    X, Y = apnea_ecg["X"], apnea_ecg["Y"]
size= len(X)
feature_set=[]
with open(r'''D:\Data_set.csv''', 'w') as f:
 f.write('std_hr,sdsd,sdnn,rmssd,range_nni,pnni_50,pnni_20,nni_50,nni_20,min_hr,median_nni,mean_nni,mean_hr,SD1, SD2,ratio_sd2_sd1,CSI,CVI,modifiedCVI,KURT,AVRR,s1,hfnu,lfnu,lf_hf_ratio,total_power,lf,hf,vlf,perm,spectral,label\n')
 for i in range (size):
  tfeature=get_time_domain_features(X[i]*1000)
  s1=skew(X[i])
  AVRR= sum(X[i])/len(X[i])
  KURT=kurtosis(X[i])
  diff_nn_intervals = np.diff(X[i])
  SD1=np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
  SD2=np.sqrt(2 * np.std(X[i], ddof=1) ** 2 - 0.5 * np.std(diff_nn_intervals, ddof=1) ** 2)
  ratio_sd2_sd1 = SD2 / SD1
  L=4 * SD1
  T=4 * SD2
  CSI= L/T
  CVI=np.log10(L * T)
  modifiedCVI= L ** 2 / T
  ffeature= get_frequency_domain_features(X[i]*1000)
  perm_ECG =  perm_entropy(X[i], order=3, normalize=True)
  f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(tfeature['std_hr'],tfeature['sdsd'],tfeature['sdnn'],tfeature['rmssd'],tfeature['range_nni'],tfeature['pnni_50'],tfeature['pnni_20'],tfeature['nni_50'],tfeature['nni_20'],tfeature['min_hr'],tfeature['median_nni'],tfeature['mean_nni'],tfeature['mean_hr'],SD1, SD2,ratio_sd2_sd1,CSI,CVI,modifiedCVI,KURT,AVRR,s1,ffeature['hfnu'],ffeature['lfnu'],ffeature['lf_hf_ratio'],ffeature['total_power'],ffeature['lf']/ffeature['total_power'],ffeature['hf']/ffeature['total_power'],ffeature['vlf'],perm_ECG,Y[i]))
