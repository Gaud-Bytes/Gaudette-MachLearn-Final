#This will look at the structure of the Heart Disease Vitals and Classifications
import sys
sys.path.append('/home/ubuntu/workspace/utils')

import mglearn as mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X = np.loadtxt('Heart_Disease_X.csv',skiprows=0, unpack=False, delimiter=',')
y = np.loadtxt('Heart_Disease_y.csv',skiprows=0, unpack=False, delimiter=',')

l = []

for i in range(len(X)):
    l.append(X[i])
    
X = np.array(l);

print(X);
# Examine Heart Disease Data to be loaded into algorithms
X_train , X_test , y_train , y_test = train_test_split(X, y, stratify=y, random_state=42) 
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))