#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:38:25 2021

@author: vittorio
"""

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder


def Encode_Data(TrainingSet, Labels):
    T_set = TrainingSet
    coordinates = T_set[:,0:2].reshape(len(T_set[:,0:2]),2)
    # encode psi
    psi = T_set[:,2].reshape(len(T_set[:,2]),1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_psi = onehot_encoder.fit_transform(psi)
    T_set = np.concatenate((coordinates,onehot_encoded_psi),1)
    Heading_set = Labels
    info = []
    info.append(onehot_encoded_psi.shape[1])
    
    return T_set, Heading_set, info

class Supervised_options_init:
    def __init__(self, env, TrainingSet):
        self.env = env
        self.TrainingSet = TrainingSet
        
    def Options(self, Noptions):
        
        stateSpace = self.env.stateSpace
        self.Noptions = Noptions
        Options = np.zeros((len(stateSpace),))
        
        if Noptions == 2:
            for i in range(len(stateSpace)):
                if stateSpace[i,1] <= 5:
                    Options[i] = 0
                else:
                    Options[i] = 1
                    
        if Noptions == 3:
            for i in range(len(stateSpace)):
                if stateSpace[i,1] <= 5:
                    Options[i] = 0
                elif stateSpace[i,1] > 5 and stateSpace[i,0] > 4:
                    Options[i] = 1
                else:
                    Options[i] = 2
                    
        if Noptions == 4:
            for i in range(len(stateSpace)):
                if stateSpace[i,1] <= 5 and stateSpace[i,0] < 5:
                    Options[i] = 0
                elif stateSpace[i,1] < 5 and stateSpace[i,0] >= 5:
                    Options[i] = 1
                elif stateSpace[i,1] >= 5 and stateSpace[i,0] > 4:
                    Options[i] = 2
                else:
                    Options[i] = 3
                    
        self.options = Options
        self.prob = F.one_hot(torch.LongTensor(Options), num_classes=Noptions)
        
        Options_set = np.zeros((len(self.TrainingSet),))
        for i in range(len(self.TrainingSet)):
            index = self.env.FindStateIndex(self.TrainingSet[i,:])
            Options_set[i] = self.options[index]
            
        return Options_set
    
    def Options_alternative(self, Noptions):
        
        self.Noptions = Noptions
        Options = np.zeros((len(self.TrainingSet),))
        
        if Noptions == 2:
            for i in range(len(self.TrainingSet)):
                if self.TrainingSet[i,2] <= 1:
                    Options[i] = 0
                else:
                    Options[i] = 1
                    
        if Noptions == 3:
            for i in range(len(self.TrainingSet)):
                if self.TrainingSet[i,2] <= 1:
                    Options[i] = 0
                elif self.TrainingSet[i,2] == 2:
                    Options[i] = 1
                else:
                    Options[i] = 2
                    
        if Noptions == 4:
            for i in range(len(self.TrainingSet)):
                if self.TrainingSet[i,2] == 0:
                    Options[i] = 0
                elif self.TrainingSet[i,2] == 1:
                    Options[i] = 1
                elif self.TrainingSet[i,2] == 2:
                    Options[i] = 2
                else:
                    Options[i] = 3
                    
        self.options_alternative = Options
        self.prob = F.one_hot(torch.LongTensor(Options), num_classes=Noptions)
            
        return Options
    
    def Plot_init(self):
        mapsize = self.env.map.shape
        #count walls
        nwalls=0;
        walls = np.empty((0,2),int)
        for i in range(0,mapsize[0]):
            for j in range(0,mapsize[1]):
                if self.env.map[i,j]==self.env.WALL:
                    walls = np.append(walls, [[j, i]], 0)
                    nwalls += 1
                    
        clrs = sns.color_palette("husl", self.Noptions)
                    
        plt.figure()
        plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')

        for i in range(0,nwalls):
            plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                     [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
    
        for i in range(0,nwalls):
            plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                     [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
            
        for s in range(0,len(self.options)):
            plt.text(self.env.stateSpace[s,1]+0.3, self.env.stateSpace[s,0]+0.4, str(int(self.options[s])+1), color='k', 
                     bbox = dict(boxstyle = "square", facecolor = clrs[int(self.options[s])]))
            
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.draw()        
            
    def Plot_alternative(self):
        
        for l in range(self.Noptions):
            mapsize = self.env.map.shape
            #count walls
            nwalls=0;
            walls = np.empty((0,2),int)
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.env.map[i,j]==self.env.WALL:
                        walls = np.append(walls, [[j, i]], 0)
                        nwalls += 1
                        
            clrs = sns.color_palette("husl", self.Noptions)
                        
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    
            for i in range(0,nwalls):
                plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
        
            for i in range(0,nwalls):
                plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                
            for s in range(0,len(self.options_alternative)):
                if int(self.TrainingSet[s,2])==l:
                    plt.text(self.TrainingSet[s,1]+0.3, self.TrainingSet[s,0]+0.4, str(int(self.options_alternative[s])+1), color='k', 
                             bbox = dict(boxstyle = "square", facecolor = clrs[int(self.options_alternative[s])]))
                
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.draw()    

