#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:06:19 2020

@author: juan
"""

#Regresion con random fores

#Importo las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importo el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

"""
#Divido el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
"""
#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

#Ajusto la regresion con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor( n_estimators = 160, criterion = 'mse',random_state = 0)
regression.fit(X, y)

#Prediccion de nuestros modelos
value = np.array( [ [2], [4], [6.5] ] )     #Valores a predecir
y_predicted = regression.predict(value)

print("El valor predecido para X = ", value, " es = ", y_predicted)


#Visualizacion de los resultados del modelo
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red', label = 'Data')
plt.plot(X_grid, regression.predict(X_grid), color = 'green', label = 'Prediction')
plt.title("Modelo de regresion por Random Forest")
plt.xlabel("Datos eje X")
plt.ylabel("Datos eje Y")
plt.legend()
plt.show()

