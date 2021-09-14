#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:00:09 2021

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

from evaluation import HierarchicalStochasticSampleTrajMDP
from evaluation import eval_policy

import TRPO
import GAIL
import PPO
import UATRPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
def IL(env, args, seed):
    
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
        "M_step_epoch": args.maximization_epochs_IL,
        "batch_size": args.batch_size_IL,
        "l_rate": args.l_rate_IL,
        "encoding_info": encoding_info
        }
    
    Agent_BatchHIL_torch = BatchBW(**kwargs)

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
    np.save(f"./results/FlatRL/IL_{args.env}_{seed}", evaluation_HIL)
    Agent_BatchHIL_torch.save(f"./models/FlatRL/IL/IL_{args.env}_{seed}")
    
    
def RL(env, args, seed):
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Rotation = np.load("./Expert_data/Rotation.npy", allow_pickle=True).tolist()
    TrainingSet = Trajectories[args.coins]
    Labels = Rotation[args.coins]
    state_samples, action_samples, encoding_info = Encode_Data(TrainingSet, Labels)
    state_dim = state_samples.shape[1]
    action_dim = env.action_size
    
    # Initialize policy        
    if args.policy == "TRPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "encoding_info": encoding_info,
         "num_steps_per_rollout": args.number_steps_per_iter
         }
         # Target policy smoothing is scaled wrt the action scale
        policy = TRPO.TRPO(**kwargs)
        if args.load_model and args.IL:
        	policy.load_actor(f"./models/FlatRL/IL/IL_{args.env}_{seed}", HIL=args.IL) 
     
      # Initialize policy        
    if args.policy == "UATRPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "encoding_info": encoding_info,
         "num_steps_per_rollout": args.number_steps_per_iter
          }
          # Target policy smoothing is scaled wrt the action scale
        policy = UATRPO.UATRPO(**kwargs)
        if args.load_model and args.IL:
        	policy.load_actor(f"./models/FlatRL/IL/IL_{args.env}_{seed}", HIL=args.IL) 
         
    if args.policy == "PPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "encoding_info": encoding_info,
         "num_steps_per_rollout": args.number_steps_per_iter
        }
          # Target policy smoothing is scaled wrt the action scale
        policy = PPO.PPO(**kwargs)
        if args.load_model and args.IL:
        	policy.load_actor(f"./models/FlatRL/IL/IL_{args.env}_{seed}", HIL=args.IL) 
         
    if args.GAIL:
        kwargs = {
          "state_dim": state_dim,
          "action_dim": action_dim,
          "expert_states": state_samples,
          "expert_actions": action_samples,
          }
        IRL = GAIL.Gail(**kwargs)
         	
     # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, seed, 0)]
    
    for i in range(int(args.max_iter)):
        
        if args.GAIL and args.Mixed_GAIL:
            rollout_states, rollout_actions = policy.GAE(env, args.GAIL, IRL.discriminator, 'standard', TrainingSet[0,:], args.Mixed_GAIL)
            mean_expert_score, mean_learner_score = IRL.update(rollout_states, rollout_actions)
            print(f"Expert Score: {mean_expert_score}, Learner Score: {mean_learner_score}")
            policy.train(Entropy = True)
            
        elif args.GAIL:
            rollout_states, rollout_actions = policy.GAE(env, args.GAIL, IRL.discriminator, 'standard', TrainingSet[0,:])
            mean_expert_score, mean_learner_score = IRL.update(rollout_states, rollout_actions)
            print(f"Expert Score: {mean_expert_score}, Learner Score: {mean_learner_score}")
            policy.train(Entropy = True)
            
        else:
            rollout_states, rollout_actions = policy.GAE(env)
            policy.train(Entropy = True)
             
        # Evaluate episode
        if (i + 1) % args.eval_freq == 0:
             evaluations.append(eval_policy(policy, env, seed, i+1, args.evaluation_episodes, 'standard', TrainingSet[0,:]))             
             
    return evaluations, policy
    
    
def train(env, args, seed): 
    
    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.IL:
        IL(env, args, seed)
        
    evaluations, policy = RL(env, args, seed)
        
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
    parser.add_argument("--number_options", default=1, type=int)     # number of options
    parser.add_argument("--policy", default="UATRPO")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=21, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env", default="Foraging")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=30000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=200, type=int)    # Max time steps to run environment
    parser.add_argument("--coins", default=2, type=int)
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--Nprocessors", default=int(0.5*multiprocessing.cpu_count()), type=int)
    #IL
    parser.add_argument("--IL", default=True, type=bool)         # Batch size for HIL
    parser.add_argument("--size_data_set", default=3000, type=int)         # Batch size for HIL
    parser.add_argument("--batch_size_IL", default=32, type=int)         # Batch size for HIL
    parser.add_argument("--maximization_epochs_IL", default=10, type=int) # Optimization epochs HIL
    parser.add_argument("--l_rate_IL", default=0.001, type=float)         # Optimization epochs HIL
    parser.add_argument("--N_iterations", default=11, type=int)            # Number of EM iterations
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
    parser.add_argument("--evaluation_max_n_steps", default = mean_len_trajs, type=int)
    args = parser.parse_args()
    
    if args.multiprocessing:   
        file_name = f"{args.policy}_IL_{args.IL}_GAIL_{args.GAIL}_Mixed_{args.Mixed_GAIL}_{args.env}_{args.Nprocessors}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, IL: {args.IL}, GAIL: {args.GAIL}, Mixed: {args.Mixed_GAIL}, Env: {args.env}, NSeeds: {args.Nprocessors}")
        print("---------------------------------------")
        
    else:
        file_name = f"{args.policy}_IL_{args.IL}_GAIL_{args.GAIL}_Mixed_{args.Mixed_GAIL}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, IL: {args.IL}, GAIL: {args.GAIL}, Mixed: {args.Mixed_GAIL}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
       
    if not os.path.exists("./results/FlatRL"):
        os.makedirs("./results/FlatRL")
               
    if not os.path.exists(f"./models/FlatRL/{file_name}"):
        os.makedirs(f"./models/FlatRL/{file_name}")
        
    if not os.path.exists("./models/FlatRL/IL"):
        os.makedirs("./models/FlatRL/IL")
        
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
        
        np.save(f"./results/FlatRL/mean_{file_name}", np.mean(evaluations,0))
        np.save(f"./results/FlatRL/std_{file_name}", np.std(evaluations,0))
        np.save(f"./results/FlatRL/steps_{file_name}", np.linspace(0, args.max_iter*args.number_steps_per_iter, len(np.mean(evaluations,0))))
        
        if args.save_model: 
            index = np.argmax(np.max(evaluations,1))
            policy = results[index][1]
            policy.save(f"./models/FlatRL/{file_name}")
    else:
        evaluations, policy = train(env, args, args.seed)
        if args.save_model: 
            np.save(f"./results/FlatRL/evaluation_{file_name}", evaluations)
            policy.save_actor(f"./models/FlatRL/{file_name}")
    