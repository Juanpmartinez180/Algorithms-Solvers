
#XGboost
#GRADIENT BOOSTING 

"""It's a technic to enhance the ML models
"""
#--------PREPROCESSING ----------
#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset import
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#categorical data codification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:,1])
X[:, 2] = labelencoder_X2.fit_transform(X[:,2])

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
                       remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

#Dataset split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#----------MODEL SELECTION-------
#XGboost model training
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


