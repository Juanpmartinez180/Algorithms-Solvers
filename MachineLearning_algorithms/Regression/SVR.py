
#Support Vector Regression (SVR)

#Importo las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importo el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#Divido el dataset en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_pred = sc_X.fit_transform(X)
y_pred = sc_y.fit_transform(y.reshape(-1,1))


#Ajusto la regresion con el dataset
from sklearn.svm import SVR     #Importo las libreria de SVM
regression = SVR(kernel='rbf')              #Creo el objeto regression
regression.fit(X_pred, y_pred)


#Prediccion de nuestros modelos con SVR
value = np.array( [ [2], [4], [6.5] ] )     #Valores a predecir

y_predicted = regression.predict( sc_X.transform(value) )   #Calculo prediccion del modelo

y_predicted = sc_y.inverse_transform( y_predicted ) #Invierto el escalado de la prediccion
print("El valor predecido para X = ", value, " es = ", y_predicted)


#Visualizacion de los resultados del modelo

X_grid = np.arange(min(X_pred), max(X_pred), 0.1)       #Creo valores intermedios del dataset
X_grid = X_grid.reshape(len(X_grid), 1) 

y_grid = regression.predict(X_grid)         #Prediccion de los valores de la grid

X_grid = sc_X.inverse_transform(X_grid)     #Invierto el escalado de X_grid
y_grid = sc_y.inverse_transform(y_grid)     #Invierto el escalado de y_grid

plt.scatter(X, y, color = 'red', label='Datos')
plt.plot(X_grid, y_grid, color = 'blue', label='Predicción')

plt.title("Modelo de regresion SVR")
plt.xlabel("Datos eje X")
plt.ylabel("Datos eje Y")
plt.legend()
plt.show()

