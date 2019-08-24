#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:05:33 2019

@author: mohityadav
"""

import pandas as pd
import numpy as np

np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.head()

#Splitting into training and test dataset
df_features = df_train.iloc[:,1:785]
df_label = df_train.iloc[:,0]
X_test = df_test.iloc[:,0:784]

from sklearn.model_selection import train_test_split
X_train,X_cv, y_train, y_cv = train_test_split(df_features,df_label,test_size=0.2,random_state = 1212)
X_train = X_train.as_matrix().reshape(33600,784)
X_cv = X_cv.as_matrix().reshape(8400, 784)
X_test = X_test.as_matrix().reshape(28000, 784)

print((min(X_train[1]), max(X_train[1])))

#Normalization and One Hot Encoded
X_train = X_train.astype('float32');X_cv = X_cv.astype('float32');X_test =X_test.astype('float32')

X_train /= 255;X_cv /= 255; X_test /= 255

#convert labels to one hot encoded
num_digits = 10
y_train = keras.utils.to_categorical(y_train, num_digits)
y_cv = keras.utils.to_categorical(y_cv, num_digits)

#
