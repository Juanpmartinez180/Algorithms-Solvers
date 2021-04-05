# -*- coding: utf-8 -*-

#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#--------------Data preprocessing--------------
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13]

#Categorical data codification
"""The idea is convert all the categorical data in to numbers via codification process.
In this case, the X set contain 1 column with countries and other with gender.
The column 1 is going to be splited in to 3 columns with 0 or 1 doing reference to the categories replaced
The column 2 as well is replaced by o or 1 depending the gender (categories)
In the first codification(column1) we need to eliminate y column due data redundany.
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()    
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])    #Encode 1 categorial variables
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X1.fit_transform(X[:, 2])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]    #Eliminate 1 column to prevent data redundancy

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



#---------- ANN build ----------
import keras
from keras.models import Sequential
from keras.layers import Dense

#ANN init.
classifier = Sequential()
#Entry and hidden layers
classifier.add( Dense(units=6, kernel_initializer= 'uniform', activation = 'relu', input_dim = 11) )

classifier.add( Dense(units=6, kernel_initializer= 'uniform', activation = 'relu' ) )

classifier.add( Dense(units=1, kernel_initializer= 'uniform', activation = 'sigmoid' ) )

#ANN compile
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#ANN training 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100 )

#ANN prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #Umbral de eleccion

#-------Model evaluation ------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

result = cm
TP = result[1,1]    #True positive
TN = result[0,0]    #True negative
FP = result[1,0]    #False positive
FN = result[0,1]    #False negative

accuracy = (TP+TN) / (np.sum(result))
pressition = (TP)/(TP+FP)
recall = (TP)/(TP+FN)
F1_score = (2*pressition*recall) / (pressition+recall)




