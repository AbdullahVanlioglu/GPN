from models.generator import Generator
from agents.agent import Agent
from trainer import Trainer
from game.env import Env
# from rl_agent import rl
import torch


def main(game_name, game_length):
	#Game description
	env = Env(game_name, game_length)

	#Network
	latent_shape = (512,)
	dropout = 0
	lr = .0001
	gen = Generator(latent_shape, env, 'nearest', dropout, lr)

	#Agent
	num_processes = 1
	experiment = "Experiments"
	lr = .00025
	model = 'base'
	dropout = .3
	reconstruct = None
	r_weight = .05
	Agent.num_steps = 5
	Agent.entropy_coef = .01
	Agent.value_loss_coef = .1
	agent = Agent(env, num_processes, experiment, 0, lr, model, dropout, reconstruct, r_weight)
	#agent = rl(env)

	#Training
	gen_updates = 1e2
	gen_batch = 32
	gen_batches = 1
	diversity_batches = 0
	rl_batch = 1e2
	pretrain = 0
	elite_persist = False
	elite_mode = 'mean'
	load_version = 0
	notes = ''
	agent.writer.add_hparams({'Experiment': experiment, 'RL_LR':lr, 'Minibatch':gen_batch, 'RL_Steps': rl_batch, 'Notes':notes}, {})
	t = Trainer(gen, agent, experiment, load_version, elite_mode, elite_persist)
	t.loss = lambda x, y: x.mean().pow(2)
	t.train(gen_updates, gen_batch, gen_batches, diversity_batches, rl_batch, pretrain)

if(__name__ == "__main__"):
	main('gpn', 1000)