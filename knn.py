# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:19:22 2020

@author: sharm
"""

import pandas as pd

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import scale
import sklearn as sk
import pickle
data = pd.read_csv("wbcd.csv")
data.head()
data.isnull().sum()
X = data.loc[:,['radius_mean','area_mean','smoothness_mean']]
y = data.loc[:,'diagnosis']
X = scale(X)
X = pd.DataFrame(X)
#X.head()
type(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
model_knn = KNeighborsClassifier(n_neighbors = 15, metric='euclidean')

model_knn.fit(X_train,y_train)
model_knn
pv = model_knn.predict(X_test)
np.mean(pv == y_test)
accuracy_score(pv,y_test)
confusion_matrix(y_test,pv)
pickle.dump(model_knn,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
