
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
X[:,1]=labelencoder_x_1.fit_transform(X[:,1])
labelencoder_x_2=LabelEncoder()
X[:,2]=labelencoder_x_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#part 2 make the ANN
#import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initialising ANN
classifier=Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
classifier.add(Dropout(p=0.1))
#adding the second hidden layer
classifier.add(Dense(activation='relu',units=6, kernel_initializer='uniform'))
classifier.add(Dropout(p=0.1))
#adding output layer
classifier.add(Dense(activation='sigmoid',units=1, kernel_initializer='uniform'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#predicting the test set results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





#test whether a customer will leave a bank or not
"""following are the specs
Georgraphy=france
creditscore=600
balance=60000
no of product=2
credit card=yes
tenure=3
age=40
"""
new_prediction=classifier.predict(sc.transform(np.array([[0.0,0,600,1,400,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction > 0.5)    



#part-4 evaluating improving and tuning the ANN
#evaluating the annn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu',units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid',units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
accuracies=accuracies.mean()
variance=accuracies.std()

