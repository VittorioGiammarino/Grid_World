#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Four_Rooms:
    class Environment:
        def __init__(self, reward_coordinate = np.array([[10,10], [10,0], [0,0], [0,10]]), reward_sequence = np.array([0,1,2,3]), init_state = np.array([0,0,0]), max_episode_steps = 200):
            self.Nc = 10 #Time steps required to bring drone to base when it crashes
            self.P_WIND = 0.1 #Gust of wind probability
            self.FREE = 0
            self.WALL = 1
            
            #Actions index
            self.NORTH = 0
            self.SOUTH = 1
            self.EAST = 2
            self.WEST = 3
            self.HOVER = 4
            
            self.state = init_state
            self.init_state = init_state
            self.observation_space = np.array([len(self.state)])
            
            self.action_size = 5
            self._max_episode_steps = max_episode_steps
            self.step_counter = 0
            
            self.reward_coordinate = reward_coordinate
            self.reward_sequence = reward_sequence
            
            Four_Rooms.Environment.GenerateMap_StateSpace(self)
            Four_Rooms.Environment.ComputeTransitionProbabilityMatrix(self)
            
        def GenerateMap_StateSpace(self):    
            mapsize = [11, 11]
            grid = np.zeros((mapsize[0], mapsize[1]))
            #define obstacles
            grid[0,5] = self.WALL
            grid[2:8,5]= self.WALL
            grid[9:11,5]= self.WALL
            grid[5,0]= self.WALL
            grid[5,2:5]= self.WALL
            grid[4,6:8]= self.WALL
            grid[4,9:11]= self.WALL
                        
            self.map = grid
            
            stateSpace = np.empty((0,2),int)
            for m in range(0,self.map.shape[0]):
                for n in range(0,self.map.shape[1]):
                    if self.map[m,n] != self.WALL:
                        stateSpace = np.append(stateSpace, [[m, n]], 0)
                    
            self.stateSpace = stateSpace
                        
        def FindStateIndex(self, value):
            K = self.stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if self.stateSpace[k,0]==value[0] and self.stateSpace[k,1]==value[1]:
                    stateIndex = k
    
            return stateIndex
        
        def ComputeTransitionProbabilityMatrix(self):
            K = self.stateSpace.shape[0]
            P = np.zeros((K,K,self.action_size))
            [M,N]=self.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = Four_Rooms.Environment.FindStateIndex(self,array_temp)

                    if self.map[i,j] != self.WALL:

                        for u in range(0,self.action_size):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == self.EAST and self.map[i,j+1]!=self.WALL:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                                elif j==N-1 and u==self.EAST:
                                    comp_no=1
                            #west case
                            if j!=0:
                                if u==self.WEST and self.map[i,j-1]!=self.WALL:
                                    r=i
                                    s=j-1
                                    comp_no=0
                                elif j==0 and u==self.WEST:
                                    comp_no=1
                            #south case
                            if i!=0:
                                if u==self.SOUTH and self.map[i-1,j]!=self.WALL:
                                    r=i-1
                                    s=j
                                    comp_no=0
                                elif i==0 and u==self.SOUTH:
                                    comp_no=1
                            #north case
                            if i!=M-1:
                                if u==self.NORTH and self.map[i+1,j]!=self.WALL:
                                    r=i+1
                                    s=j
                                    comp_no=0
                                elif i==M-1 and u==self.NORTH:
                                    comp_no=1
                            #hover case
                            if u==self.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t = Four_Rooms.Environment.FindStateIndex(self, array_temp)
 
                                # No wind case
                                P[k,t,u] = P[k,t,u]+(1-self.P_WIND)
                                base0 = Four_Rooms.Environment.FindStateIndex(self, self.init_state)

                                # case wind

                                #north wind
                                if s+1>N-1 or self.map[r,s+1]==self.WALL:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t = Four_Rooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #north wind no hit

                                #South Wind
                                if s-1<0 or self.map[r,s-1]==self.WALL:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=Four_Rooms.Environment.FindStateIndex(self,array_temp)                                 
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #south wind no hit

                                #East Wind
                                if r+1>M-1 or self.map[r+1,s]==self.WALL:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=Four_Rooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #east wind no hit

                                #West Wind
                                if r-1<0 or self.map[r-1,s]==self.WALL:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=Four_Rooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #west wind no hit

                            elif comp_no == 1:
                                base0=base0 = Four_Rooms.Environment.FindStateIndex(self, self.init_state)
                                P[k,base0,u]=1

            self.P = P
            
            
        def PlotMap(self):
            mapsize = self.map.shape
            #count walls
            nwalls=0;
            walls = np.empty((0,2),int)
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.map[i,j]==self.WALL:
                        walls = np.append(walls, [[j, i]], 0)
                        nwalls += 1
                        
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    
            for i in range(0,nwalls):
                plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
        
            for i in range(0,nwalls):
                plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.draw()
                
                  
        def seed(self, seed):
            self.seed = seed
            np.random.seed(self.seed)
                
        def reset(self, version = 'standard', init_state = np.array([0,0,0])):
            if version == 'standard':
                self.state = init_state
                self.step_counter = 0
                self.reward_counter = 0
                self.active_reward = int(self.state[2])
            else:
                mapsize = self.map.shape()
                for i in range(30):
                    init_state = np.random.randint(0,mapsize[1],2)
                    if self.map[init_state] == self.FREE:
                        break
                self.state = np.append(init_state,0)
                self.step_counter = 0
                self.reward_counter = 0
                self.active_reward = int(self.state[2])
                
            return self.state
                    
        def random_sample(self):
            return np.random.randint(0,self.action_size)       
        
        def step(self, action):
            
            self.step_counter +=1
            r=0
            
            # given action, draw next state
            state_index = Four_Rooms.Environment.FindStateIndex(self, self.state[0:2])
            x_k_possible=np.where(self.P[state_index,:,int(action)]!=0)
            prob = self.P[state_index,x_k_possible[0][:],int(action)]
            prob_rescaled = np.divide(prob,np.amin(prob))

            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            state_plus1 = self.stateSpace[x_k_possible[0][index_x_plus1],:]
            
            if state_plus1[0] == self.reward_coordinate[self.active_reward][0] and state_plus1[1] == self.reward_coordinate[self.active_reward][1]:
                r=r+1
                if self.reward_counter < len(self.reward_sequence)-1:
                    self.reward_counter += 1
                else:
                    self.reward_counter = 0
                    
                self.active_reward = self.reward_sequence[self.reward_counter]
                
            self.state = np.append(state_plus1, self.active_reward)
            if self.step_counter >= self._max_episode_steps:
                done = True
            else:
                done = False            
            
            return self.state, r, done, False  
      
    class Optimal_Expert:
        def __init__(self):
            self.Environment = Four_Rooms.Environment()
            self.nReward = len(self.Environment.reward_coordinate)
            self.R_indexes = []
            for i in range(self.nReward):
                self.R_indexes.append(self.Environment.FindStateIndex(self.Environment.reward_coordinate[i]))
                
            self.P = self.Environment.P
        
        def ComputeStageCosts(self):
            
            Costs = []
            action_space = self.Environment.action_size
            for z in range(self.nReward):
                
                K = self.Environment.stateSpace.shape[0]
                G = np.zeros((K,action_space))
                [M,N]=self.Environment.map.shape
    
                for i in range(0,M):
                    for j in range(0,N):
        
                        array_temp = [i, j]
                        k = self.Environment.FindStateIndex(array_temp)
        
                        if self.Environment.map[i,j] != self.Environment.WALL:
        
                            if k == self.R_indexes[z]:
                                dummy=0 #no cost
                            else:
                                for u in range(0,action_space):
                                    comp_no=1;
                                    # east case
                                    if j!=N-1:
                                        if u == self.Environment.EAST and self.Environment.map[i,j+1]!=self.Environment.WALL:
                                            r=i
                                            s=j+1
                                            comp_no = 0
                                    elif j==N-1 and u==self.Environment.EAST:
                                        comp_no=1
        
                                    if u == self.Environment.EAST:
                                        if j==N-1 or self.Environment.map[i,j+1]==self.Environment.WALL:
                                            G[k,u]=np.inf
        
                                    #west case
                                    if j!=0:
                                        if u==self.Environment.WEST and self.Environment.map[i,j-1]!=self.Environment.WALL:
                                            r=i
                                            s=j-1
                                            comp_no=0
                                    elif j==0 and u==self.Environment.WEST:
                                        comp_no=1
        
                                    if u==self.Environment.WEST:
                                        if j==0 or self.Environment.map[i,j-1]==self.Environment.WALL:
                                            G[k,u]=np.inf
        
                                    #south case
                                    if i!=0:
                                        if u==self.Environment.SOUTH and self.Environment.map[i-1,j]!=self.Environment.WALL:
                                            r=i-1
                                            s=j
                                            comp_no=0
                                    elif i==0 and u==self.Environment.SOUTH:
                                        comp_no=1
        
                                    if u==self.Environment.SOUTH:
                                        if i==0 or self.Environment.map[i-1,j]==self.Environment.WALL:
                                            G[k,u]=np.inf
        
                                    #north case
                                    if i!=M-1:
                                        if u==self.Environment.NORTH and self.Environment.map[i+1,j]!=self.Environment.WALL:
                                            r=i+1
                                            s=j
                                            comp_no=0
                                    elif i==M-1 and u==self.Environment.NORTH:
                                        comp_no=1
        
                                    if u==self.Environment.NORTH:
                                        if i==M-1 or self.Environment.map[i+1,j]==self.Environment.WALL:
                                            G[k,u]=np.inf
        
                                    #hover case
                                    if u==self.Environment.HOVER:
                                        r=i
                                        s=j
                                        comp_no=0
        
                                    if comp_no==0:
                                        array_temp = [r, s]
        
                                        G[k,u] = G[k,u]+(1-self.Environment.P_WIND) #no shot and no wind
        
                                        # case wind
        
                                        #north wind
                                        if s+1>N-1 or self.Environment.map[r,s+1]==self.Environment.WALL:
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                        else:
                                            array_temp = [r, s+1]
        
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25
        
        
                                        #South Wind
                                        if s-1<0 or self.Environment.map[r,s-1]==self.Environment.WALL:
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                        else:
                                            array_temp = [r, s-1]
        
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #south wind no hit
        
                                        #East Wind
                                        if r+1>M-1 or self.Environment.map[r+1,s]==self.Environment.WALL:
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                        else:
                                            array_temp = [r+1, s]
        
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #east wind no hit
        
                                        #West Wind
                                        if r-1<0 or self.Environment.map[r-1,s]==self.Environment.WALL:
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                        else:
                                            array_temp = [r-1, s]
                                        
                                            G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #west wind no hit
        
                                    elif comp_no == 1:
                                        dummy=0
        
                for l in range(0,action_space):
                    G[self.R_indexes[z],l]=0
                    
                Costs.append(G)
    
            self.Costs = Costs          
        
        def ValueIteration(self):
            
            action_space = self.Environment.action_size
            tol=10**(-5)
            optimal_policy = []
            optimal_value = []
            
            for z in range(self.nReward):
                G = self.Costs[z]
                Terminal_index = self.R_indexes[z]
                K = G.shape[0]
                V=np.zeros((K,action_space))
                VV=np.zeros((K,2))
                I=np.zeros((K))
                Err=np.zeros((K))
        
                #initialization
                VV[:,0]=50
                VV[Terminal_index,0]=0
                n=0
                Check_err=1
        
                while Check_err==1:
                    n=n+1
                    Check_err=0
                    for k in range(0,K):
                        if n>1:
                            VV[:,0]=VV[0:,1]
        
                        if k==Terminal_index:
                            VV[k,1]=0
                            V[k,:]=0
                        else:
                            CTG=np.zeros((action_space)) #cost to go
                            for u in range(0,action_space):
                                for j in range(0,K):
                                    CTG[u]=CTG[u] + self.P[k,j,u]*VV[j,1]
        
                                V[k,u]=G[k,u]+CTG[u]
        
                            VV[k,1]=np.amin(V[k,:])
                            flag = np.where(V[k,:]==np.amin(V[k,:]))
                            I[k]=flag[0][0]
        
                        Err[k]=abs(VV[k,1]-VV[k,0])
        
                        if Err[k]>tol:
                            Check_err=1
        
                J_opt=VV[:,1]
                I[Terminal_index]=self.Environment.HOVER
                u_opt = I[:]
                
                optimal_policy.append(u_opt)
                optimal_value.append(J_opt)
    
            self.optimal_policy = optimal_policy
            self.optimal_value = optimal_value
                
        def PlotPolicy(self):
            
            for z in range(self.nReward):
                
                plt.figure()
                mapsize = self.Environment.map.shape
                nwalls=0;
                walls = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if self.Environment.map[i,j]==self.Environment.WALL:
                            walls = np.append(walls, [[j, i]], 0)
                            nwalls += 1        
                # Plot
                plt.figure()
                plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
        
                for i in range(0,nwalls):
                    plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
            
                for i in range(0,nwalls):
                    plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                    
                    
                u = self.optimal_policy[z]
                for s in range(0,u.shape[0]):
                    if u[s] == self.Environment.NORTH:
                        txt = u'\u2191'
                    elif u[s] == self.Environment.SOUTH:
                        txt = u'\u2193'
                    elif u[s] == self.Environment.EAST:
                        txt = u'\u2192'
                    elif u[s] == self.Environment.WEST:
                        txt = u'\u2190'
                    elif u[s] == self.Environment.HOVER:
                        txt = u'\u2715'
                    plt.text(self.Environment.stateSpace[s,1]+0.3, self.Environment.stateSpace[s,0]+0.5, txt)
                    
                plt.gca().set_aspect('equal', adjustable='box')
                plt.axis('off')
                plt.draw()
                
                
    class Simulation:
        def __init__(self):
            self.Environment = Four_Rooms.Environment()
            self.nReward = len(self.Environment.reward_coordinate)
            self.R_indexes = []
            for i in range(self.nReward):
                self.R_indexes.append(self.Environment.FindStateIndex(self.Environment.reward_coordinate[i]))
                
            self.P = self.Environment.P
            
            Expert = Four_Rooms.Optimal_Expert()
            Expert.ComputeStageCosts()
            Expert.ValueIteration()
            
            self.policy = Expert.optimal_policy
            
        def SampleTrajs(self, seed, number_of_trajectories = 10):
            self.Environment.seed(seed)
            size_input = self.Environment.observation_space[0]
            
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_location = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
            
            for t in range(number_of_trajectories):
                x = np.empty((0, size_input))
                reward_location = np.empty((0, size_input-1))
                u_tot = np.empty((0,0),int)
                cum_reward = 0 
                state, done = self.Environment.reset(version='standard'), False
                while not done:
                    r_location = self.Environment.reward_coordinate[state[2]]
                    reward_location = np.append(reward_location, r_location.reshape(1, len(r_location)), 0)
                    policy = self.policy[state[2]]
                    state_index = self.Environment.FindStateIndex(state[0:2])
                    action = policy[state_index]
                    x = np.append(x, state.reshape(1, size_input), 0)
                    u_tot = np.append(u_tot, action)    
                    state, reward, done, _ = self.Environment.step(action)
                    cum_reward += reward
                    
                traj[t] = x
                control[t]=u_tot
                Reward_location[t] = reward_location
                Reward_array = np.append(Reward_array, cum_reward)
                
            return traj, control, Reward_array, Reward_location  
        
        def VideoSimulation(self, u, states, reward_location, name_video):
            
            mapsize = self.Environment.map.shape
            nwalls=0;
            walls = np.empty((0,2),int)
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.Environment.map[i,j]==self.Environment.WALL:
                        walls = np.append(walls, [[j, i]], 0)
                        nwalls += 1        
            # Plot
            fig = plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    
            for i in range(0,nwalls):
                plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
        
            for i in range(0,nwalls):
                plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.draw()

            ims = []
            for s in range(0,len(u)):
                if u[s] == self.Environment.NORTH:
                    txt = u'\u2191'
                elif u[s] == self.Environment.SOUTH:
                    txt = u'\u2193'
                elif u[s] == self.Environment.EAST:
                    txt = u'\u2192'
                elif u[s] == self.Environment.WEST:
                    txt = u'\u2190'
                elif u[s] == self.Environment.HOVER:
                    txt = u'\u2715'    
                im1 = plt.text(states[s,1]+0.0, states[s,0]+0.0, txt, fontsize=20)
                im2 = plt.text(reward_location[s,1]+0.0, reward_location[s,0]+0.0, 'R', fontsize=20, color = 'r')
                ims.append([im1, im2])
    
            ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                            repeat_delay=2000)
            
            if not os.path.exists("./Videos/Optimal_Expert"):
                os.makedirs("./Videos/Optimal_Expert")
            
            ani.save("./Videos/Optimal_Expert/" + name_video)
                    
                    
                
                

            
                

    
                 
            
            
            
            
            
            
            
            
            
            