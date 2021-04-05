#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:06:43 2020

@author: juan
"""

#Modelo de regresion lineal simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3 , random_state = 0)

#Crear modelo de regresion lineal con el conjunto de entrenamient
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los resultados del entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.scatter(X_test, y_test, color = "green" )
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Training set)")
plt.xlabel("Experiencia [Años]")
plt.ylabel("Sueldo [$]")
plt.show()

#Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red" )
plt.plot(X_test, regression.predict(X_test), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Training set)")
plt.xlabel("Experiencia [Años]")
plt.ylabel("Sueldo [$]")
plt.show()





