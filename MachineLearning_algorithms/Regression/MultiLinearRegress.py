#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:56:31 2020

@author: juan
"""

#Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer( [ ('one_hot_encoder', OneHotEncoder(categories='auto'), [3]) ],
                       remainder = 'passthrough' )
X = np.array(ct.fit_transform(X),dtype = np.float)

#Evitar la trampa de las variables ficticias
X = X[:, 1:]

#Dividimos el dataset en training set y testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )

#Ajustar el modelo de regresion lineal multiple con el training set
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de los resultados en el test set
y_pred = regression.predict(X_test)

#Construir el modeo óptimo de RLM utilizando la eliminacion hacia atrás
import statsmodels.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1) #Añado una columna a la matriz de datos, para representar el termino independiente b0 de la RLM
SL = 0.05   #Nivel de significancia

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]        #Elimine la 2 columna ya que tenia el mayor P valor
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]       #Elimine la 2 columna del X_opt anterior (tenia mayor p valor)
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0, 3, 5]]         #Elimie la 3 columna del X_opt anterior 
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0, 3]]            #Elimine la 3 columna del X_opt anterior
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
print(regression_OLS.summary())



