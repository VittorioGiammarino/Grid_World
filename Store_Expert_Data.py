#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 14:33:43 2021

@author: vittorio
"""
import os
import numpy as np

import World

env = World.Four_Rooms.Environment()
env.PlotMap()
Expert = World.Four_Rooms.Optimal_Expert()
Expert.ComputeStageCosts()
Expert.ValueIteration()
Expert.PlotPolicy()

sim = World.Four_Rooms.Simulation()
seed = 0
traj, control, Reward_array, Reward_location = sim.SampleTrajs(seed)
sim.VideoSimulation(control[0], traj[0], Reward_location[0], "Expert_video.mp4")

TrainingSet = np.array([item for sublist in traj for item in sublist])
Labels = np.array([item for sublist in control for item in sublist])


if not os.path.exists("./Expert_data"):
    os.makedirs("./Expert_data")
    
np.save("./Expert_data/TrainingSet", TrainingSet)
np.save("./Expert_data/Labels", Labels)
np.save("./Expert_data/Reward", Reward_array)




