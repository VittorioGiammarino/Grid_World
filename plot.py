#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:36:58 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %% Load Data

TrainingSet = np.load("./Expert_data/TrainingSet.npy")
Labels = np.load("./Expert_data/Labels.npy")
Reward = np.load("./Expert_data/Reward.npy")

Expert_reward_mean = np.mean(Reward)
Expert_reward_std = np.std(Reward)

# %%

initialization = ["random", "supervised", "supervised_alternative"]
env = "Four_Rooms"
number_options = [1,2,3,4]
number_options_supervised = [2,3,4]
seed_trial = [0,1,2,3,4,5,6,7]

results_random = [[] for option in number_options]
results_supervised = [[] for option in number_options_supervised]
results_supervised_alternative = [[] for option in number_options_supervised]



init = "random"
for option in number_options:
    for seed in seed_trial:
        results_random[option-number_options[0]].append(np.load(f"./results/IL_Noptions_{option}_initialization_{init}_{env}_Seed_{seed}.npy"))

steps = np.linspace(len(TrainingSet),len(TrainingSet)*len(results_random[0][0]),len(results_random[0][0]))
goal = Expert_reward_mean*np.ones(len(steps))
std = Expert_reward_std*np.ones(len(steps))
clrs = sns.color_palette("husl", 12)


fig, ax = plt.subplots()
ax.plot(steps, goal, label="Expert", c=clrs[11])
for i in range(len(results_random)):
    temp = np.array(results_random[i])
    temp_mean = np.mean(temp,0)
    temp_std = np.std(temp,0)
    ax.plot(steps, temp_mean, label=f"IL_Noptions_{i+1}_init_{init}", c=clrs[i])
    # ax.fill_between(steps, temp_mean-temp_std, temp_mean+temp_std, alpha=0.2, facecolor=clrs[i])
    
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title(f'{init} Init HIL')
            
init = "supervised"
for option in number_options_supervised:
    for seed in seed_trial:
        results_supervised[option-number_options_supervised[0]].append(np.load(f"./results/IL_Noptions_{option}_initialization_{init}_{env}_Seed_{seed}.npy"))
        
steps = np.linspace(len(TrainingSet),len(TrainingSet)*len(results_random[0][0]),len(results_random[0][0]))
fig, ax = plt.subplots()
ax.plot(steps, goal, label="Expert", c=clrs[11])
for i in range(len(results_supervised)):
    temp = np.array(results_supervised[i])
    temp_mean = np.mean(temp,0)
    temp_std = np.std(temp,0)
    ax.plot(steps, temp_mean, label=f"IL_Noptions_{i+2}_init_{init}", c=clrs[i])
    # ax.fill_between(steps, temp_mean-temp_std, temp_mean+temp_std, alpha=0.2, facecolor=clrs[i])
    
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title(f'{init} Init HIL')


init = "supervised_alternative"          
for option in number_options_supervised:
    for seed in seed_trial:
        results_supervised_alternative[option-number_options_supervised[0]].append(np.load(f"./results/IL_Noptions_{option}_initialization_{init}_{env}_Seed_{seed}.npy"))          
            
steps = np.linspace(len(TrainingSet),len(TrainingSet)*len(results_random[0][0]),len(results_random[0][0]))
fig, ax = plt.subplots()
ax.plot(steps, goal, label="Expert", c=clrs[11])
for i in range(len(results_supervised_alternative)):
    temp = np.array(results_supervised_alternative[i])
    temp_mean = np.mean(temp,0)
    temp_std = np.std(temp,0)
    ax.plot(steps, temp_mean, label=f"IL Noptions: {i+2}, init: {init}", c=clrs[i])
    # ax.fill_between(steps, temp_mean-temp_std, temp_mean+temp_std, alpha=0.2, facecolor=clrs[i])
    
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title(f'{init} Init HIL')


for i in range(len(results_supervised_alternative)):
    fig, ax = plt.subplots()
    ax.plot(steps, goal, label="Expert", c=clrs[11])
    init = "supervised_alternative"  
    temp = np.array(results_supervised_alternative[i])
    temp_mean = np.mean(temp,0)
    temp_std = np.std(temp,0)
    ax.plot(steps, temp_mean, label=f"IL Noptions: {i+2}, init: {init}", c=clrs[0])
    init = "supervised"
    temp = np.array(results_supervised[i])
    temp_mean = np.mean(temp,0)
    temp_std = np.std(temp,0)
    ax.plot(steps, temp_mean, label=f"IL Noptions: {i+2}, init: {init}", c=clrs[1])    
    init = "random"
    temp = np.array(results_random[i])
    temp_mean = np.mean(temp,0)
    temp_std = np.std(temp,0)
    ax.plot(steps, temp_mean, label=f"IL Noptions: {i+2}, init: {init}", c=clrs[2]) 
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.set_title(f'Noptions: {i+2}, HIL')

    

    
