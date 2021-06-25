#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

from networks import Agent
from mixer import QMixer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np

import helper
import os
from episode_memory import Memory
import random
import json

#robot_index
GK_INDEX = 0 
D1_INDEX = 1 
D2_INDEX = 2 
F1_INDEX = 3 
F2_INDEX = 4

def update_target_model(net, target_net):
    target_net.load_state_dict(net.state_dict())

class QMIX:
    def __init__(self, n_agents, n_roles, n_mixers, dim_obs, dim_globalstate, dim_act, CHECKPOINT, CHECKPOINT_MIXER, load=False, play=False):
        params_file = open(os.path.dirname(__file__) + '/parameters.json')
        params = json.loads(params_file.read())
        self.agent_id = n_agents
        self.role_type = n_roles
        self.mixer_num = n_mixers
        self.agents_GK = 1
        self.agents_D12 = 2
        self.agents_F12 = 2
        self.n_agents = [self.agents_GK, self.agents_D12, self.agents_F12]
        self.CHECKPOINT = CHECKPOINT
        self.CHECKPOINT_MIXER = CHECKPOINT_MIXER
        self._iterations = 0
        self.update_steps = 300 # Update Target Network
        self.epsilon_steps = params['qmix_parameters']['epsilon_steps'] # Decrease epsilon
        self.play = play
        if self.play == True:
            self.epsilon = 0.05 # Greedy choice if play is True
        else:
            self.epsilon = 1.0 # Initial epsilon value      
        self.final_epsilon = 0.05 # Final epsilon value
        self.dec_epsilon =  0.025 # Decrease rate of epsilon for every generation

        self.observation_steps = 100 # Number of iterations to observe before start training
        self.save_num = 500 # Save checkpoint
        self.batch_size = params['qmix_parameters']['batch_size']

        self.num_inputs = dim_obs
        self.act_size = dim_act
        self.mixer_state = dim_globalstate
        self.mixing_dim = 32
        self.memory = Memory(params['qmix_parameters']['buffer_size']) # replay buffer
        self.net = [Agent(self.num_inputs, self.act_size), 
                    Agent(self.num_inputs, self.act_size),
                    Agent(self.num_inputs, self.act_size)]
        self.target_net = deepcopy(self.net)
        self.mixer = [QMixer(self.n_agents[1], self.mixer_state, self.mixing_dim), QMixer(self.n_agents[2], self.mixer_state, self.mixing_dim)]
        self.target_mixer = deepcopy(self.mixer)
        self.load = load
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.load == True:
            for role in range(self.role_type):
                self.net[role].load_state_dict(torch.load(CHECKPOINT[role], map_location=torch.device(self.device)))
            for role in range(self.mixer_num):
                self.mixer[role].load_state_dict(torch.load(CHECKPOINT_MIXER[role], map_location=torch.device(self.device)))
            helper.printConsole("loading variables...")

        self.params_GK = list(self.net[GK_INDEX].parameters())
        self.params_D = list(self.net[1].parameters())
        self.params_F = list(self.net[2].parameters())
        self.params_D12 = self.params_D + list(self.mixer[0].parameters())
        self.params_F12 = self.params_F + list(self.mixer[1].parameters())
        self.params = [self.params_GK, self.params_D12, self.params_F12]

        for role in range(self.role_type):
            update_target_model(self.net[role], self.target_net[role])
        for role in range(self.mixer_num):
            update_target_model(self.mixer[role], self.target_mixer[role])
        for role in range(self.role_type):
            self.net[role].train()
            self.target_net[role].train() 
            self.net[role].to(self.device)
            self.target_net[role].to(self.device)
        for role in range(self.mixer_num):
            self.mixer[role].train()
            self.target_mixer[role].train()
            self.mixer[role].to(self.device)
            self.target_mixer[role].to(self.device)

        self.gamma = params['qmix_parameters']['gamma']
        self.grad_norm_clip = params['qmix_parameters']['grad_norm_clip']
        self.loss = [0 for _ in range(self.role_type)]
        self.lr = params['qmix_parameters']['lr']
        self.optimizer = [optim.Adam(params=self.params[role], lr=self.lr) for role in range(self.role_type)]

    def select_action(self, state):

        state = torch.Tensor(state).to(self.device)

        out_put_actions = []
        for role in range(self.agent_id):

            if role == 0:
                index = 0
            elif role == 1 or role == 2:
                index = 1
            else:
                index = 2
            qvalue = self.net[index](state[role])
            qvalue = qvalue.cpu().data.numpy()
        
            pick_random = int(np.random.rand() <= self.epsilon)
            random_actions = np.random.randint(0, self.act_size, 1)
            picked_actions = pick_random * random_actions + (1 - pick_random) * np.argmax(qvalue)
            out_put_actions += list(picked_actions)

        return out_put_actions, self.epsilon

    def store_experience(self, state, globalstate, next_globalstate, next_state, act, rew):
        # Store transition in the replay buffer.
        self.memory.push(state, globalstate, next_globalstate, next_state, act, rew)

    def update_policy(self):

        batch = self.memory.sample(self.batch_size)

        states = torch.Tensor(batch.state).to(self.device)
        global_states = torch.Tensor(batch.globalstate).to(self.device)
        next_global_states = torch.Tensor(batch.next_globalstate).to(self.device)
        next_states = torch.Tensor(batch.next_state).to(self.device)
        actions = torch.Tensor(batch.action).long().to(self.device)
        rewards = torch.Tensor(batch.reward).to(self.device)

        states_GK = states[:,0,:] 
        states_D1 = states[:,1,:]
        states_D2 = states[:,2,:] 
        states_F1 = states[:,3,:]
        states_F2 = states[:,4,:]
        states = [states_GK, states_D1, states_D2, states_F1, states_F2]
        next_states_GK = next_states[:,0,:] 
        next_states_D1 = next_states[:,1,:]
        next_states_D2 = next_states[:,2,:]
        next_states_F1 = next_states[:,3,:]
        next_states_F2 = next_states[:,4,:]
        next_states = [next_states_GK, next_states_D1, next_states_D2, next_states_F1, next_states_F2]
        #rewards = rewards[:, :-1]
        rewards_GK = rewards[:,0]
        rewards_D12 = (rewards[:,1] + rewards[:,2])/2
        rewards_F12 = (rewards[:,3] + rewards[:,4])/2
        rewards = [rewards_GK, rewards_D12, rewards_F12]
        #actions = actions[:, :-1]
        actions_GK = actions[:,0]
        actions_D1 = actions[:,1]
        actions_D2 = actions[:,2]
        actions_F1 = actions[:,3]
        actions_F2 = actions[:,4]
        actions = [actions_GK, actions_D1, actions_D2, actions_F1, actions_F2]
        
        for role in range(self.role_type):

            if (role == 0):
            
                q_values = self.net[role](states[role]).squeeze(1)
                max_next_q_values = self.target_net[role](next_states[role]).squeeze(1).max(1)[0]

                one_hot_action = torch.zeros(self.batch_size, q_values.size(-1)).to(self.device)
                one_hot_action.scatter_(1, actions[role].unsqueeze(1), 1)
                chosen_q_values = torch.sum(q_values.mul(one_hot_action), dim=1)

            # Mixing qvalue when agent is not GK
            elif (role == 1) or (role == 2):

                index_1 = 2*(role - 1) + 1
                index_2 = 2*(role - 1) + 2

                q_values_1 = self.net[role](states[index_1]).squeeze(1)
                max_next_q_values_1 = self.target_net[role](next_states[index_1]).squeeze(1).max(1)[0]

                one_hot_action_1 = torch.zeros(self.batch_size, q_values.size(-1)).to(self.device)
                one_hot_action_1.scatter_(1, actions[index_1].unsqueeze(1), 1)
                chosen_q_values_1 = torch.sum(q_values_1.mul(one_hot_action_1), dim=1)

                q_values_2 = self.net[role](states[index_2]).squeeze(1)
                max_next_q_values_2 = self.target_net[role](next_states[index_2]).squeeze(1).max(1)[0]

                one_hot_action_2 = torch.zeros(self.batch_size, q_values.size(-1)).to(self.device)
                one_hot_action_2.scatter_(1, actions[index_2].unsqueeze(1), 1)
                chosen_q_values_2 = torch.sum(q_values_2.mul(one_hot_action_2), dim=1)

                chosen_q_values = torch.stack([chosen_q_values_1, chosen_q_values_2], axis=1)
                max_next_q_values = torch.stack([max_next_q_values_1, max_next_q_values_2], axis=1)

                chosen_q_values = self.mixer[role-1](chosen_q_values, global_states)
                max_next_q_values = self.target_mixer[role-1](max_next_q_values, next_global_states)

            target = rewards[role] + self.gamma * max_next_q_values

            td_error = (chosen_q_values - target.detach())
            loss = (td_error ** 2).sum() 
            self.loss[role] = loss.cpu().data.numpy()
        
            self.optimizer[role].zero_grad()
            loss.backward()
            # need to solve this when using 5 agents
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net[role].parameters(), self.grad_norm_clip)
            if (role == 1) or (role == 2):
                grad_norm_2 = torch.nn.utils.clip_grad_norm_(self.mixer[role-1].parameters(), self.grad_norm_clip)
            self.optimizer[role].step()
        
        if self._iterations  % self.update_steps == 0: 
            for role in range(self.role_type):
                update_target_model(self.net[role], self.target_net[role])
            for role in range(self.mixer_num):
                update_target_model(self.mixer[role], self.target_mixer[role])
            helper.printConsole("Updated target model.")

        if self._iterations  % self.epsilon_steps == 0: 
            self.epsilon = max(self.epsilon - self.dec_epsilon, self.final_epsilon)
            helper.printConsole("New Episode! New Epsilon:" + str(self.epsilon))

        return self.loss

    def save_checkpoint(self, iteration):
        if iteration % self.save_num ==0:
            for role in range(self.role_type):
                self.net[role].save_model(self.net[role], self.CHECKPOINT[role])
            for role in range(self.mixer_num):
                torch.save(self.mixer[role].state_dict(), self.CHECKPOINT_MIXER[role])
            helper.printConsole("Saved Checkpoint.")
        if iteration % 200000 == 0: # Save checkpoint every 200000 iterations
            for role in range(self.role_type):
                name = self.CHECKPOINT[role] + str(iteration)
                self.net[role].save_model(self.net[role], name)
            for role in range(self.mixer_num):
                name = self.CHECKPOINT_MIXER[role] + str(iteration)
                torch.save(self.mixer[role].state_dict(), name)
            helper.printConsole("Saved Play Checkpoint.")

    def print_loss(self, loss, iteration):
        if self._iterations % 100 == 0: # Print information every 100 iterations
            helper.printConsole("======================================================")
            helper.printConsole("Agent: " + str(self.agent_id))
            helper.printConsole("Epsilon: " + str(self.epsilon))
            helper.printConsole("iterations: " + str(self._iterations))
            helper.printConsole("GK_Loss: " + str(loss[0]))
            helper.printConsole("D12_Loss: " + str(loss[1]))
            helper.printConsole("F12_Loss: " + str(loss[2]))
            helper.printConsole("======================================================")

    def update(self):
        if len(self.memory) > self.observation_steps:
            self._iterations += 1
            loss = self.update_policy()
            self.print_loss(loss, self._iterations)  