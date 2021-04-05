#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:37:12 2020

@author: juan
"""
#Regresion polinómica

#Importo las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importo el DataSet
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Ajusto la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Ajusto la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree = 4)
X_poly = poli_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)     #Creo nuevos valores intermedios de X para la regresion
X_grid = X_grid.reshape(len(X_grid),1)      #Convierto vector en matriz
X_poly_grid = poli_reg.fit_transform(X_grid)

#Visualizacion de los resultados del modelo lineal
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X))
plt.title("Modelo de regresion Lineal")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo estimado [$]")
plt.show()

#Visualizacion de los resultados del modelo polinómico
plt.scatter(X, y)
plt.plot(X, lin_reg_2.predict(X_poly))      #Ploteo regresion con valores del dataset
plt.plot(X_grid, lin_reg_2.predict(X_poly_grid) )     #Ploteo regresion con nuevos valores intermedios
plt.title("Modelo de regresion Polinómica")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo estimado [$]")
plt.show()

#Prediccion de nuestros modelos
print(lin_reg.predict([[6.5]]))     #Prediccion de la reg lineal para empleado clase 6.5
print(lin_reg_2.predict(poli_reg.fit_transform([[6.5]])))



