import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Buffer import ReplayBuffer
from models import SoftmaxHierarchicalActor
from models import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class H_TD0(object):
    def __init__(self, state_dim, action_dim, option_dim, termination_dim, encoding_info = None, l_rate_critic=3e-4, discount=0.99, tau=0.005, eta = 1e-7):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        self.termination_dim = termination_dim
        self.encoding_info = encoding_info
        
        self.Buffer = [[None]*1 for _ in range(option_dim)]
        self.pi_hi = SoftmaxHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
        self.Critic = Critic(state_dim, action_dim, option_dim).to(device)
        self.Critic_target = copy.deepcopy(self.Critic)
        self.pi_lo = [[None]*1 for _ in range(option_dim)]
        self.pi_b = [[None]*1 for _ in range(option_dim)]
        
        pi_lo_temp = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_dim).to(device)
        pi_b_temp = SoftmaxHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
        for option in range(option_dim):
            self.Buffer[option] = ReplayBuffer(state_dim, action_dim)
            self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b[option] = copy.deepcopy(pi_b_temp)        
            
        # define optimizer 
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=l_rate_critic)   
                    
        self.discount = discount
        self.tau = tau
        self.eta = eta
        
        self.total_it = 0
        
    def select_action(self, state, option):
        state = H_TD0.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        prob_u = self.pi_lo[option](state).cpu().data.numpy()
        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
        for i in range(1,prob_u_rescaled.shape[1]):
            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
        temp = np.where(draw_u<=prob_u_rescaled)[1]
        if temp.size == 0:
            action = np.argmax(prob_u)
        else:
            action = np.amin(np.where(draw_u<=prob_u_rescaled)[1])
        return int(action)
        
    def select_option(self, state, b, previous_option):
        state = H_TD0.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)     
        if b == 1:
            b_bool = True
        else:
            b_bool = False

        o_prob_tilde = np.empty((1,self.option_dim))
        if b_bool == True:
            o_prob_tilde = self.pi_hi(state).cpu().data.numpy()
        else:
            o_prob_tilde[0,:] = 0
            o_prob_tilde[0,previous_option] = 1

        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        temp = np.where(draw_o<=prob_o_rescaled)[1]
        if temp.size == 0:
             option = np.argmax(prob_o)
        else:
             option = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
             
        return option
    
    def select_termination(self, state, option):
        state = H_TD0.encode_state(self, state)
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)         
        self.pi_b[option].eval()
        # Termination
        prob_b = self.pi_b[option](state).cpu().data.numpy()
        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
        for i in range(1,prob_b_rescaled.shape[1]):
            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
        temp = np.where(draw_b<=prob_b_rescaled)[1]
        if temp.size == 0:
            b = np.argmax(prob_b)
        else:
            b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
            
        return int(b)  
    
    def encode_state(self, state):
        state = state.flatten()
        coordinates = state[0:2]
        psi = state[2]
        psi_encoded = np.zeros(self.encoding_info[0])
        psi_encoded[int(psi)]=1
        coin_dir_encoded = np.zeros(self.encoding_info[1])
        coin_dir = state[3]
        coin_dir_encoded[int(coin_dir)]=1
        current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
        return current_state_encoded
    
    def encode_action(self, action):
        action_encoded = np.zeros(self.action_dim)
        action_encoded[int(action)]=1
        return action_encoded

    def train(self, option, batch_size=256):
        
        self.total_it += 1
        
        # Sample replay buffer 
        state, action, next_state, reward, cost, not_done = self.Buffer[option].sample(batch_size)
        option_vector = torch.ones_like(reward[:,0] , dtype=int)
        
        with torch.no_grad():
        	# Compute the target Q value
            
            next_action_option_i = []
            log_pi_next_state_option_i = []
            for option_i in range(self.option_dim):
                next_action, log_pi_next_state = self.pi_lo[option_i].sample(next_state)
                next_action = F.one_hot(next_action, num_classes=self.action_dim)
                next_action_option_i.append(next_action)
                log_pi_next_state_option_i.append(log_pi_next_state)
            
            next_action = next_action_option_i[option]
            first_term_target_Q1, first_term_target_Q2 = self.Critic_target(next_state, next_action, F.one_hot(option*option_vector, num_classes=self.option_dim))
            pi_b = self.pi_b[option](next_state).cpu()
            target_Q1 = pi_b[:,0].reshape(-1,1)*first_term_target_Q1
            target_Q2 = pi_b[:,0].reshape(-1,1)*first_term_target_Q2
            
            for option_i in range(self.option_dim):
                next_action = next_action_option_i[option_i]
                second_term_target_Q1, second_term_target_Q2 = self.Critic_target(next_state, next_action, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                target_Q1 += pi_b[:,1].reshape(-1,1)*self.pi_hi(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q1
                target_Q2 += pi_b[:,1].reshape(-1,1)*self.pi_hi(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q2
                 
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward-cost + not_done * self.discount * target_Q
            
		# Get current Q estimates
        current_Q1, current_Q2 = self.Critic(state, action, F.one_hot(option*option_vector, num_classes=self.option_dim))
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
		# Update the frozen target models
        for param, target_param in zip(self.Critic.parameters(), self.Critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
    def save(self, filename):
        torch.save(self.Critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")  
    
    def load(self, filename):
        self.pi_hi.load_state_dict(torch.load(filename + "_pi_hi"))
        
        for option in range(self.option_dim):
            self.pi_lo[option].load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
            self.pi_b[option].load_state_dict(torch.load(filename + f"_pi_b_option_{option}"))