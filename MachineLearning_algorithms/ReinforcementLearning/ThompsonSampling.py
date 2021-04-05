#Thompson Sampling Algorithm

#Library Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset import
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Algorithm
N = np.size(dataset, 0) #Round Number
d = np.size(dataset, 1) #Options Number
N1 = np.zeros((d))   #Number of rewards = 1
N0 = np.zeros((d))   #Number of rewards = 0 
ads_selected = []
total_reward = 0    

for n in range(0, N):
    max_random = 0  #Variable a optimizar
    ad = 0
    for i in range(0, d):
        random_beta = np.random.beta( N1[i] + 1, N0[i] + 1 )
        if random_beta > max_random : 
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        N1[ad] += 1 
    else :
        N0[ad] += 1
    total_reward += reward
    
#Results visualization
plt.hist(ads_selected)
plt.title('Thompson Sampling Algorithm')
plt.xlabel('Option')
plt.ylabel('Frecuency')

    