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


#Applying Neural Network to Model with 4 layers (300,100,100,200)
n_input = 784
n_hidden_1 = 300;
n_hidden_2 = 100;
n_hidden_3 = 100;
n_hidden_4 = 200;
num_digits = 10;

Inp = Input(shape=(784,))
x = Dense(units = n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(units = n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(units = n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(units = n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(units = num_digits, activation='softmax', name = "Output_Layer")(x)

model = Model(Inp, output)
model.summary()

#Insert Hyperparameters
learning_rate = 0.1
training_epochs = 20
batch_size = 100
sgd= optimizers.SGD(lr = learning_rate)

#Using Stochastic gradient descent and compiling a model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics = ['accuracy'])

#Fitting model
history1 = model.fit(X_train, y_train, 
                     batch_size= batch_size,
                     epochs= training_epochs,
                     verbose= 2,
                     validation_data=(X_cv, y_cv))


Inp = Input(shape=(784,))
x = Dense(units = n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(units = n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(units = n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(units = n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(units = num_digits, activation='softmax', name = "Output_Layer")(x)

#Now using ADAM as optimizer and compiling a model
adam = keras.optimizers.Adam(lr = learning_rate)
model2 = Model(Inp, output)
model2.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

#Fitting model
history2 = model2.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=training_epochs,
                      verbose=2,
                      validation_data=(X_cv, y_cv))

#Changing Learning rate to 0.01 or 0.5
Inp = Input(shape=(784,))
x = Dense(units = n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(units = n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(units = n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(units = n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(units = num_digits, activation='softmax', name = "Output_Layer")(x)

learning_rate = 0.01
adam = keras.optimizers.Adam(lr = learning_rate)
model2a = Model(Inp, output)
model2a.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])













