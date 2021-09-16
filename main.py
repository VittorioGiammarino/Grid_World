#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:29:20 2021

@author: vittorio
"""
import torch
import argparse
import os
import numpy as np
import multiprocessing as mp
import multiprocessing.pool

import World
from utils import Supervised_options_init
from utils import Encode_Data
from BatchBW_HIL_torch import BatchBW

from evaluation import HierarchicalStochasticSampleTrajMDP
from evaluation import eval_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

def IL(env, args, seed):
    
    TrainingSet = np.load("./Expert_data/TrainingSet.npy", allow_pickle=True)
    Labels = np.load("./Expert_data/Labels.npy", allow_pickle=True)
    
    state_samples, action_samples, encoding_info = Encode_Data(TrainingSet[0:args.size_data_set,:], Labels[0:args.size_data_set])
        
    state_dim = state_samples.shape[1]
    action_dim = env.action_size
    option_dim = args.number_options
    termination_dim = 2
    
    kwargs = {
    	"state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "state_samples": state_samples,
        "action_samples": action_samples,
        "M_step_epoch": args.maximization_epochs_IL,
        "batch_size": args.batch_size_IL,
        "l_rate": args.l_rate_IL,
        "encoding_info": encoding_info
        }
    
    Agent_BatchHIL_torch = BatchBW(**kwargs)
    
    if args.initialization == "supervised":
        supervisor = Supervised_options_init(env, TrainingSet[0:args.size_data_set,:])
        epochs = args.init_supervised_epochs
        Options = supervisor.Options(args.number_options)
        Agent_BatchHIL_torch.pretrain_pi_hi(epochs, Options)
        Labels_b = Agent_BatchHIL_torch.prepare_labels_pretrain_pi_b(Options)
        for i in range(args.number_options):
            Agent_BatchHIL_torch.pretrain_pi_b(epochs, Labels_b[i], i)
            
    if args.initialization == "supervised_alternative":
        supervisor = Supervised_options_init(env, TrainingSet[0:args.size_data_set,:])
        epochs = args.init_supervised_epochs
        Options = supervisor.Options_alternative(args.number_options)
        Agent_BatchHIL_torch.pretrain_pi_hi(epochs, Options)
        Labels_b = Agent_BatchHIL_torch.prepare_labels_pretrain_pi_b(Options)
        for i in range(args.number_options):
            Agent_BatchHIL_torch.pretrain_pi_b(epochs, Labels_b[i], i)
    
    Loss = 100000
    evaluation_HIL = []
    for i in range(args.N_iterations):
        print(f"Iteration {i+1}/{args.N_iterations}")
        loss = Agent_BatchHIL_torch.Baum_Welch()
        if loss > Loss:
            Agent_BatchHIL_torch.reset_learning_rate(args.l_rate_IL/10)
        Loss = loss
        [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
         TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_BatchHIL_torch, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        avg_reward = np.sum(RewardBatch_torch)/args.evaluation_episodes
        evaluation_HIL.append(avg_reward)
        
        print("---------------------------------------")
        print(f"Seed {seed}, Evaluation over {args.evaluation_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        
    # Save
    if not os.path.exists(f"./models/IL_Noptions_{args.number_options}_initialization_{args.initialization}_{args.env}_Seed_{seed}"):
        os.makedirs(f"./models/IL_Noptions_{args.number_options}_initialization_{args.initialization}_{args.env}_Seed_{seed}")
    
    np.save(f"./results/IL_Noptions_{args.number_options}_initialization_{args.initialization}_{args.env}_Seed_{seed}", evaluation_HIL)
    Agent_BatchHIL_torch.save(f"./models/IL_Noptions_{args.number_options}_initialization_{args.initialization}_{args.env}_Seed_{seed}/IL_")
    
def train(env, args, seed): 
    
    # Set seeds
    env.Seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.IL:
        IL(env, args, seed)
        
    # return evaluations, policy


if __name__ == "__main__":
    
    TrainingSet = np.load("./Expert_data/TrainingSet.npy", allow_pickle=True)
    Labels = np.load("./Expert_data/Labels.npy", allow_pickle=True)
    env = World.Four_Rooms.Environment()
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--number_options", default=4, type=int)     # number of options
    parser.add_argument("--initialization", default="random")     # random, supervised, supervised_alternative
    parser.add_argument("--init_supervised_epochs", default=200, type=int)    # supervised
    parser.add_argument("--policy", default="UATRPO")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env", default="Four_Rooms")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=10*env._max_episode_steps, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (iters) we evaluate
    parser.add_argument("--max_iter", default=200, type=int)    # Max iteration to run training
    #IL
    parser.add_argument("--IL", default=True, type=bool)         # Batch size for HIL
    parser.add_argument("--size_data_set", default=2000, type=int)   
    parser.add_argument("--batch_size_IL", default=32, type=int)         # Batch size for HIL
    parser.add_argument("--maximization_epochs_IL", default=10, type=int) # Optimization epochs HIL
    parser.add_argument("--l_rate_IL", default=0.001, type=float)         # Optimization epochs HIL
    parser.add_argument("--N_iterations", default=10, type=int)            # Number of EM iterations
    # IRL
    parser.add_argument("--GAIL", default=False)                     # Frequency of delayed critic updates
    parser.add_argument("--Mixed_GAIL", default=False)  
    # HRL
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)                  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_max_n_steps", default = env._max_episode_steps, type=int)
    args = parser.parse_args()
    
    init = ["supervised", "supervised_alternative"]
    
    for initialization in init:
        args.initialization = initialization
        for options in range(2,5):
            args.number_options = options
            
            # if args.number_options==1 and args.initialization == "supervised":
            #     break
            # elif args.number_options==1 and args.initialization == "supervised_alternative":
            #     break
    
            file_name = f"IL_{args.IL}_options_{args.number_options}_initialization_{args.initialization}_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"IL: {args.IL}, Noptions: {args.number_options}, Initialization: {args.initialization}, Env: {args.env}, Seed: {args.seed}")
            print("---------------------------------------")
                
            if not os.path.exists("./results"):
                os.makedirs("./results")
                       
            if not os.path.exists(f"./models"):
                os.makedirs(f"./models")
                        
            # evaluations, policy = train(env, args, args.seed)
            
            train(env, args, args.seed)
            
            # if args.save_model: 
            #     np.save(f"./results/evaluation_{file_name}", evaluations)
            #     policy.save_actor(f"./models/{file_name}")

