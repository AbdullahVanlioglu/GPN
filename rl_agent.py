import os
import torch
import time
import argparse
import glob
import re
import pickle
import numpy as np
import csv
import numpy as np
from tensorboardX import SummaryWriter

from agents import dqn_model

from point_mass_formation import AgentFormation
from read_maps import fa_regenate
from models.classifier import Classifier


def parameters():
    # model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    # load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    model_dir = './saved_models'
    load_model_dir = './models'

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--device', default=device, help='device')
    parser.add_argument('--visualization', default=False, type=bool, help='number of training episodes')
    # test
    #parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
    parser.add_argument('--load_model', default=load_model_dir, help='number of training episodes')
    #parser.add_argument('--test_iteration', default=25, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    #parser.add_argument('--test_model_no', default=0, help='single model to evaluate')
    #parser.add_argument('--test_model_level', default="easy", help='single model level to evaluate')
    # training
    #parser.add_argument('--num_episodes', default=1000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=32, help='number of steps to update the policy')
    #parser.add_argument('--eval_interval', type=int, default=32, help='number of steps to eval the policy')
    parser.add_argument('--start_step', type=int, default=128, help='After how many steps to start training')
    # model
    #parser.add_argument('--resume', default=False, action='store_true', help='to continue the training')
    parser.add_argument('--model_dir', default=model_dir, help='folder to save models')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epsilon', default=0.9, type=float, help='greedy policy')
    parser.add_argument('--gamma', default=0.99, type=float, help='reward discount')
    parser.add_argument('--target_update', default=48, type=int, help='target update freq')
    parser.add_argument('--n_actions', type=int, default=8, help='number of actions (agents to produce)')
    #parser.add_argument('--n_states', type=int, default=7350, help='Number of states after convolution layer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size to train')
    parser.add_argument('--memory_size', type=int, default=250000, help='Buffer memory size')
    parser.add_argument('--multi_step', type=int, default=1, help='Multi step')
    #parser.add_argument('--out_shape', type=int, default=10, help='Observation image shape')
    parser.add_argument('--hid_size', type=int, default=64, help='Hidden size dimension')
    parser.add_argument('--out_shape_list', type=list, default=[20,40,60,80], help='output shape array')
    parser.add_argument('--fc1_shape_list', type=list, default=[16,576,1936,4096], help='fc1 size array')

    return parser.parse_args()

class rl:
    def __init__(self, env):
        self.args = parameters()
        self.iteration = 0
        self.best_reward = -np.inf
        
        # Create environments.
        self.env = AgentFormation(visualization=self.args.visualization)

        #create RL agent
        self.dqn = dqn_model.DQN(self.args),

        self.writer = SummaryWriter()

        #env
        self.env_def = env

    def train(self, fake_map, iteration):
        self.iteration = iteration
        # print("fake map", fake_map)
        ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list = fa_regenate(fake_map)

        #reset environment
        self.env.reset(ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list)

        #get action
        action = self.dqn.choose_action(agent_obs) # output is between 0 and 7
        n_agents = action + 1 # number of allowable agents is 1 to 8

        episode_reward, done, agent_next_obs = self.env.step(n_agents)

        if self.args.visualization:
            self.env.close()        
        
        self.dqn.memory.append(agent_obs, action, episode_reward, agent_next_obs, done)
        if  self.iteration > self.args.start_step and self.iteration % self.args.update_interval == 0:
            self.dqn.learn()
            #print("dqn learn")
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.dqn.save_models()

        self.writer.add_scalar('Reward', episode_reward)
        self.writer.add_scalar('Num. Agents', n_agents)
        self.writer.add_scalar('Iteration', self.iteration)

        #print(f'Train Scale- {current_scale} | Iteration: {self.iteration} | Episode Reward: {round(episode_reward, 2)}')
        #print("rl train func ended")
        
        return episode_reward

    def classify(self, fake_map, classifier):
        ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list = fa_regenate(fake_map)
        classifier.optimizer.zero_grad()
        #reset environment

        # best_agent = np.random.randint(8)
        
        min_reward = 0
        best_agent = None
        
        for i in range(8): 
            
            self.env.reset(ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list)
            n_agents = i+1
            nagent_episode_reward, _, _ = self.env.step(n_agents)

            if nagent_episode_reward > min_reward:
                best_agent = n_agents
            print(i, nagent_episode_reward)
            
        self.env.reset(ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list)
        episode_reward, done, agent_next_obs = self.env.step(best_agent)

        high_feature, output = classifier.forward(agent_obs) # 2x80x80
        output = output.unsqueeze(0)

        target = torch.ones(1).type(torch.LongTensor)*best_agent
        loss = classifier.loss(output, target)
        loss.backward()
        classifier.optimizer.step()

        self.writer.add_scalar('classifier/loss', loss.item())

        return best_agent
