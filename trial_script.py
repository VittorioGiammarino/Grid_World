#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:31:50 2021

@author: vittorio
"""
import World
from utils import Supervised_options_init
from BatchBW_HIL_torch import BatchBW
import numpy as np

env = World.Four_Rooms.Environment()
TrainingSet = np.load("./Expert_data/TrainingSet.npy", allow_pickle=True)
Labels = np.load("./Expert_data/Labels.npy", allow_pickle=True)

supervision = Supervised_options_init(env, TrainingSet)
Noptions = 4
supervision.Options(Noptions)
supervision.Plot_init()
supervision.Options_alternative(Noptions)
supervision.Plot_alternative()

# %%

env = World.Four_Rooms.Environment()
TrainingSet = np.load("./Expert_data/TrainingSet.npy", allow_pickle=True)
Labels = np.load("./Expert_data/Labels.npy", allow_pickle=True)

kwargs = {
	"state_dim": 2,
    "action_dim": 5,
    "option_dim": Noptions,
    "termination_dim": 2,
    "state_samples": TrainingSet,
    "action_samples": Labels,
    "M_step_epoch": 50,
    "batch_size": 32,
    "l_rate": 0.01,
    }

Agent_BatchHIL_torch = BatchBW(**kwargs)


supervisor = Supervised_options_init(env, TrainingSet)
epochs = 200
Options = supervisor.Options(Noptions)
Agent_BatchHIL_torch.pretrain_pi_hi(epochs, Options)

# %%
Labels_b = Agent_BatchHIL_torch.prepare_labels_pretrain_pi_b(Options)
for i in range(Noptions):
    Agent_BatchHIL_torch.pretrain_pi_b(epochs, Labels_b[i], i)


