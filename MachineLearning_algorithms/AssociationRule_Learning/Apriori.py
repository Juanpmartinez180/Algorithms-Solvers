#A priori - Asociation model

#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset import
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
columns = np.size(dataset, 0)
rows = np.size(dataset, 1)
for i in range (0, columns):
    transactions.append([str(dataset.values[i,j]) for j in range(0,rows)])

#A priori algorithm training
from apyori import apriori
"""
    min_support = considero que se tienen que comprar como minimo 3 articulos por dia
                    por 7 dias a la semana..sobre la cantidad total de usuarios o transactions
                    3[articulos] x 7[dias] / 7500[users] = aprox 0.003
    min_confidence = 0.2 valor obtenido empiricamente
"""
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length=2 )

#Results visualization
results = list(rules)

 results[3]