# -*- coding: utf-8 -*-
#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset import
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Upper confidence Bound (UCB) algorithm
d = np.size(dataset, 1) #Total options available
N = np.size(dataset, 0) #Total rounds or tests
number_of_selections = np.zeros((1, d)) #Inicializo vector a cero
sums_of_rewards = np.zeros((1,d))
ads_selected = []
total_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (number_of_selections[0,i] > 0 ):
            average_reward = sums_of_rewards[0,i] / number_of_selections[0,i]
            delta_i = np.sqrt( (3/2)*np.log(n+1) / number_of_selections[0,i] )  
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 10**400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[0, ad] = number_of_selections[0, ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[0, ad] = sums_of_rewards[0, ad] + reward
    total_reward = total_reward + reward

#Results visualization
plt.hist(ads_selected)
plt.title('Upper Confidence Bound Histogram')
plt.xlabel('Test')
plt.ylabel('Reward')
    