#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
import torch
import argparse
import os
import numpy as np
import multiprocessing as mp
import multiprocessing.pool

import World

from utils import Encode_Data
from BatchBW_HIL_torch import BatchBW

from evaluation import evaluate_H

import H_PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def HIL(env, args, seed):
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
    
    TrainingSet = Trajectories[args.coins]
    Labels = Rotation[args.coins]
   
    state_samples, action_samples, encoding_info = Encode_Data(TrainingSet, Labels)
    
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
        "M_step_epoch": args.maximization_epochs_HIL,
        "batch_size": args.batch_size_HIL,
        "l_rate": args.l_rate_HIL,
        "encoding_info": encoding_info
        }
    
    Agent_BatchHIL_torch = BatchBW(**kwargs)
    
    if args.pi_hi_supervised:
        epochs = args.pi_hi_supervised_epochs
        Options = state_samples[:,2]
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
            Agent_BatchHIL_torch.reset_learning_rate(args.l_rate_HIL/10)
        Loss = loss
        avg_reward = evaluate_H(seed, Agent_BatchHIL_torch, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
        evaluation_HIL.append(avg_reward)
        
    # Save
    np.save(f"./results/HRL/HIL_{args.env}_{seed}", evaluation_HIL)
    Agent_BatchHIL_torch.save(f"./models/HRL/HIL/HIL_{args.env}_{seed}")
    
    
def HRL(env, args, seed):
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
    TrainingSet = Trajectories[args.coins]
    Labels = Rotation[args.coins]
    state_samples, action_samples, encoding_info = Encode_Data(TrainingSet, Labels)
    state_dim = state_samples.shape[1]
    action_dim = env.action_size
    option_dim = args.number_options
    termination_dim = 2
    
    # # Initialize policy        
    # if args.policy == "HTRPO":
    #     kwargs = {
    #      "state_dim": state_dim,
    #      "action_dim": action_dim,
    #      "encoding_info": encoding_info,
    #      "num_steps_per_rollout": args.number_steps_per_iter
    #      }
    #      # Target policy smoothing is scaled wrt the action scale
    #     policy = TRPO.TRPO(**kwargs)
    #     if args.load_model and args.IL:
    #     	policy.load_actor(f"./models/FlatRL/IL/IL_{args.env}_{seed}", HIL=args.IL) 
     
    #   # Initialize policy        
    # if args.policy == "HUATRPO":
    #     kwargs = {
    #      "state_dim": state_dim,
    #      "action_dim": action_dim,
    #      "encoding_info": encoding_info,
    #      "num_steps_per_rollout": args.number_steps_per_iter
    #       }
    #       # Target policy smoothing is scaled wrt the action scale
    #     policy = UATRPO.UATRPO(**kwargs)
    #     if args.load_model and args.IL:
    #     	policy.load_actor(f"./models/FlatRL/IL/IL_{args.env}_{seed}", HIL=args.IL) 
         
    if args.policy == "HPPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "option_dim": option_dim,
         "termination_dim": termination_dim,
         "encoding_info": encoding_info,
         "num_steps_per_rollout": args.number_steps_per_iter
        }
          # Target policy smoothing is scaled wrt the action scale
        Agent_HRL = H_PPO.H_PPO(**kwargs)
        if args.load_model and args.HIL:
        	Agent_HRL.load_actor(f"./models/HRL/HIL/HIL_{args.env}_{seed}") 
         
    # if args.HGAIL:
    #     kwargs = {
    #       "state_dim": state_dim,
    #       "action_dim": action_dim,
    #       "expert_states": state_samples,
    #       "expert_actions": action_samples,
    #       }
    #     HIRL = GAIL.Gail(**kwargs)
         	
    # Evaluate untrained policy
    evaluation_HRL = []
    avg_reward = evaluate_H(seed, Agent_HRL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
    evaluation_HRL.append(avg_reward)
    
    
    for i in range(int(args.max_iter)):
        
        # if args.HGAIL and args.Mixed_HGAIL:
        #     rollout_states, rollout_actions = Agent_HRL.GAE(env, args.HGAIL, HIRL.discriminator, 'standard', TrainingSet[0,:], args.Mixed_HGAIL)
        #     mean_expert_score, mean_learner_score = HIRL.update(rollout_states, rollout_actions)
        #     print(f"Expert Score: {mean_expert_score}, Learner Score: {mean_learner_score}")
        #     policy.train(Entropy = True)
            
        # elif args.HGAIL:
        #     rollout_states, rollout_actions = Agent_HRL.GAE(env, args.HGAIL, HIRL.discriminator, 'standard', TrainingSet[0,:])
        #     mean_expert_score, mean_learner_score = HIRL.update(rollout_states, rollout_actions)
        #     print(f"Expert Score: {mean_expert_score}, Learner Score: {mean_learner_score}")
        #     policy.train(Entropy = True)
            
        # else:
        rollout_states, rollout_actions, rollout_options = Agent_HRL.GAE(env)
        Agent_HRL.train(Entropy = True)
             
        # Evaluate episode
        if (i + 1) % args.eval_freq == 0:
            avg_reward = evaluate_H(seed, Agent_HRL, env, args.evaluation_max_n_steps, args.evaluation_episodes, 'standard', TrainingSet[0,:])
            evaluation_HRL.append(avg_reward)      
             
    return evaluation_HRL, Agent_HRL
    
    
def train(env, args, seed): 
    
    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.HIL:
        HIL(env, args, seed)
        
    evaluations, policy = HRL(env, args, seed)
        
    return evaluations, policy


if __name__ == "__main__":
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
    Coins_location = np.load("./Expert_data/Coins_location.npy")
    len_trajs = []
    for i in range(len(Trajectories)):
        len_trajs.append(len(Trajectories[i]))
        
    mean_len_trajs = int(np.mean(len_trajs))
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--number_options", default=2, type=int)     # number of options
    parser.add_argument("--policy", default="HPPO")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=10, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env", default="Foraging")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=30000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=200, type=int)    # Max time steps to run environment
    parser.add_argument("--coins", default=2, type=int)
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--Nprocessors", default=int(0.5*multiprocessing.cpu_count()), type=int)
    #IL
    parser.add_argument("--HIL", default=True, type=bool)         # Batch size for HIL
    parser.add_argument("--size_data_set", default=3000, type=int)         # Batch size for HIL
    parser.add_argument("--batch_size_HIL", default=32, type=int)         # Batch size for HIL
    parser.add_argument("--maximization_epochs_HIL", default=10, type=int) # Optimization epochs HIL
    parser.add_argument("--l_rate_HIL", default=0.001, type=float)         # Optimization epochs HIL
    parser.add_argument("--N_iterations", default=11, type=int)            # Number of EM iterations
    parser.add_argument("--pi_hi_supervised", default=True, type=bool)     # Supervised pi_hi
    parser.add_argument("--pi_hi_supervised_epochs", default=200, type=int)  
    # IRL
    parser.add_argument("--HGAIL", default=False)                     # Frequency of delayed critic updates
    parser.add_argument("--Mixed_HGAIL", default=False)  
    # HRL
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)                  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_max_n_steps", default = mean_len_trajs, type=int)
    args = parser.parse_args()
    
    if args.multiprocessing:   
        file_name = f"{args.policy}_HIL_{args.HIL}_HGAIL_{args.HGAIL}_Mixed_{args.Mixed_HGAIL}_{args.env}_{args.Nprocessors}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, HIL: {args.HIL}, HGAIL: {args.HGAIL}, Mixed: {args.Mixed_HGAIL}, Env: {args.env}, NSeeds: {args.Nprocessors}")
        print("---------------------------------------")
        
    else:
        file_name = f"{args.policy}_HIL_{args.HIL}_HGAIL_{args.HGAIL}_Mixed_{args.Mixed_HGAIL}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, HIL: {args.HIL}, HGAIL: {args.HGAIL}, Mixed: {args.Mixed_HGAIL}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
       
    if not os.path.exists("./results/HRL"):
        os.makedirs("./results/HRL")
               
    if not os.path.exists(f"./models/HRL/{file_name}"):
        os.makedirs(f"./models/HRL/{file_name}")
        
    if not os.path.exists("./models/HRL/HIL"):
        os.makedirs("./models/HRL/HIL")
        
    coins_location = Coins_location[args.coins,:,:] 
    
    env = World.Foraging.env(coins_location)
    ctx = mp.get_context('spawn')
    
    if args.multiprocessing: 
        arguments = [(env, args, seed) for seed in range(args.Nprocessors)] 
        with ctx.Pool(args.Nprocessors) as pool:
            results = pool.starmap(train, arguments)
            pool.close()
            pool.join()
            
        evaluations = []
        for i in range(args.Nseed):
            evaluations.append(results[i][0])
        
        np.save(f"./results/HRL/mean_{file_name}", np.mean(evaluations,0))
        np.save(f"./results/HRL/std_{file_name}", np.std(evaluations,0))
        np.save(f"./results/HRL/steps_{file_name}", np.linspace(0, args.max_iter*args.number_steps_per_iter, len(np.mean(evaluations,0))))
        
        if args.save_model: 
            index = np.argmax(np.max(evaluations,1))
            policy = results[index][1]
            policy.save(f"./models/HRL/{file_name}")
    else:
        evaluations, policy = train(env, args, args.seed)
        if args.save_model: 
            np.save(f"./results/HRL/evaluation_{file_name}", evaluations)
            policy.save_actor(f"./models/HRL/{file_name}")
    

