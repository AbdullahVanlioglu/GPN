import os
import csv
import pathlib
import tempfile
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
import level_visualizer

import distributionLoss

import pdb

class Trainer(object):
    def __init__(self, gen, enc,  clasx, agent, save, version=0, elite_mode='max', elite_persist=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = gen.to(self.device)
        self.classifier = clasx.to(self.device)
        self.encoder = enc.to(self.device)
        self.gen_optimizer = gen.optimizer
        self.agent = agent
        self.loss = F.mse_loss #lambda x, y: (x.mean() - 0).pow(2) + (x.std() - .3).pow(2) #distributionLoss.NormalDivLoss().to(self.device)
        self.temp_dir = tempfile.TemporaryDirectory()

        self.save_paths = {'dir':save}
        self.save_paths['agent'] = os.path.join(save,'agents')
        self.save_paths['models'] = os.path.join(save,'models')
        self.save_paths['levels'] = os.path.join(save,'levels.csv')
        self.save_paths['loss'] = os.path.join(save,'losses.csv')

        #Elite Settings
        self.elite_mode = elite_mode
        self.elite_persist = elite_persist

        #Ensure directories exist
        pathlib.Path(self.save_paths['agent']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_paths['models']).mkdir(parents=True, exist_ok=True)

        if(version > 0):
            print("load version")
            self.load(version)
        else:
            self.version = 0

    def load(self, version):
        self.version = version
        self.agent.load(self.save_paths['agent'], version)

        path = os.path.join(self.save_paths['models'], "checkpoint_{}.tar".format(version))
        if(os.path.isfile(path)):
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['generator_model'])
            self.gen_optimizer.load_state_dict(checkpoint['generator_optimizer'])

    def save_models(self, version, g_loss):
        self.agent.save(self.save_paths['agent'], version)
        torch.save({
            'generator_model': self.generator.state_dict(),
            'generator_optimizer': self.gen_optimizer.state_dict(),
            'version': version,
            'gen_loss': g_loss,
            }, os.path.join(self.save_paths['models'], "checkpoint_{}.tar".format(version)))

    def save_loss(self, update, gen_loss):
        add_header = not os.path.exists(self.save_paths['loss'])
        with open(self.save_paths['loss'], 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                header = ['update', 'gen_loss']
                writer.writerow(header)
            writer.writerow((update, gen_loss))

    def save_levels(self, update, strings, rewards, expected_rewards):
        add_header = not os.path.exists(self.save_paths['levels'])
        with open(self.save_paths['levels'], 'a+') as results:
            writer = csv.writer(results)
            if(add_header):
                header = ['update', 'level', 'reward', 'expected_reward']
                writer.writerow(header)
            for i in range(len(strings)):
                writer.writerow((update, strings[i], rewards[i], expected_rewards[i].item()))

    def new_elite_levels(self, z):
        
        lvl_tensors = self.generator.new(z)
        lvl_strs = self.agent.env_def.create_levels(lvl_tensors)
        
        return lvl_tensors, lvl_strs

    def new_levels(self, z, save=False):
        lvl_tensors = self.generator.new(z)
        
        return lvl_tensors

    def freeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_weights(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def z_generator(self, batch_size, z_size):
        return lambda b=batch_size, z=z_size:torch.Tensor(b, z).normal_().to(self.device)

    def critic(self, x):
        self.agent.agent.optimizer.zero_grad()
        rnn_hxs = torch.zeros(x.size(0), self.agent.actor_critic.recurrent_hidden_state_size).to(self.device)
        masks = torch.ones(x.size(0), 1).to(self.device)
        #actions = torch.zeros_like(masks).long()

        #value, _, _, _, dist_entropy, _ = self.agent.actor_critic.evaluate_actions(x, rnn_hxs, masks, actions)
        Qs, actor_features, _ = self.agent.actor_critic.base(x, rnn_hxs, masks)
        dist = self.agent.actor_critic.dist(actor_features)
        value = (dist.probs*Qs).sum(1).unsqueeze(1)
        dist_entropy = dist.entropy().mean()
        return value, dist_entropy, actor_features
        #return self.agent.actor_critic.get_value(x, rnn_hxs, masks)

    def eval_levels(self, tensor):
        #raise Exception("Not implemented")
        #levels = self.game.create_levels(tensor)
        #What to pass to play?
        #File Names?
        #Create new envs for evaluation...
        rewards = self.agent.play(levels)
        return rewards

    def train(self, updates, batch_size, gen_updates):
        self.generator.train()
        z = self.z_generator(batch_size, self.generator.z_size) # 32x512

        loss = 0
        entropy = 0
        gen_updates = 0
        
        for update in range(int(updates)):
            lvl_tensors, lvl_strs = self.new_elite_levels(z(batch_size)) # 32x2x80x80

            for i in range(len(lvl_strs)):
                n_agents = self.agent.classify(lvl_strs[i], self.classifier)


        generated_levels = []
        reward_list = []
        for i in range(gen_updates):
            self.gen_optimizer.zero_grad()
            noise = z()
            lvl_tensors, lvl_strs = self.new_elite_levels(z(1))

            with torch.no_grad():
                output = self.classifier.forward(lvl_strs)

            n_agents = torch.argmax(output)

            ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list = self.agent.fa_regenate(lvl_strs)
            self.env.reset(ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list)
            episode_reward, _, _ = self.env.step(n_agents)
            
            reward_list.append(episode_reward)

            target = torch.zeros_like(reward_list) #was ones like
            gen_loss = self.loss(reward_list, target)

            self.agent.writer.add_scalar('generator/loss', gen_loss.item(), gen_updates)
            self.agent.writer.add_scalar('generator/loss', gen_loss.item(), gen_updates)
            # self.agent.writer.add_scalar('generator/entropy', dist.item(), gen_updates)
            # self.agent.writer.add_scalar('generator/diversity', diversity.item(), gen_updates)

            gen_updates += 1

        self.agent.writer.add_images('Generated Levels', generated_levels, (update-1), dataformats='HWC')
        #Save a generated level
        levels, states = self.new_levels(z(1)) #scale debug
        with torch.no_grad():
            expected_rewards = self.critic(states)
        #real_rewards = self.eval_levels(levels)
        real_rewards = ['Nan']
        self.save_levels(update, levels, real_rewards, expected_rewards)

        #Save and report results
        loss += gen_loss.item()
        entropy += dist.item()
        self.version += 1
        save_frequency = 100
        if(update%save_frequency == 0):
            self.save_models(update, gen_loss)
            self.save_loss(update, loss/save_frequency)
            print('[{}] Gen Loss: {}, Entropy {}'.format(update, loss/save_frequency, entropy/save_frequency))
            loss = 0
            entropy = 0
        self.agent.envs.close()
