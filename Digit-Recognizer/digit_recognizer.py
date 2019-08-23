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