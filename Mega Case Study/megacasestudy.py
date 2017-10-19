#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:51:14 2017

@author: aakashwadhwa
"""


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#training the SOM
from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration=100)

#visualizing the results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()  

#finding the frauds

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(3,5)], mappings[(1,6)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#part 2  going from unsupervised to supervised deep learning

#create the matrix of features
customers = dataset.iloc[:,1:].values

#create the dependent variables
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
#part 2 make an ann
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


#part 2 make the ANN
#import keras library and packages
from keras.models import Sequential
from keras.layers import Dense

#initialising ANN
classifier=Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(activation='relu', input_dim=15, units=2, kernel_initializer='uniform'))

#adding output layer
classifier.add(Dense(activation='sigmoid',units=1, kernel_initializer='uniform'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to the training set
classifier.fit(customers,is_fraud,batch_size=1,epochs=2)

#predicting the test set results
y_pred = classifier.predict(customers)        
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1) 
y_pred = y_pred[y_pred[:,1].argsort()]