import argparse
import gym
import time

from probe.probe_handler import ProbeHandler
from encoders.rand_cnn import RandCNN 
from atariari.benchmark.wrapper import AtariARIWrapper
from atariari.benchmark.episodes import get_episodes

# ordering for state variables in dictionary form
game_mappings = {'Pong-v0': {'player_y': 0, 'enemy_y': 1, 'ball_x': 2, 'ball_y': 3, 'enemy_score': 4, 'player_score': 5}}

def full_pipeline(args):
	# setup environment
	env = AtariARIWrapper(gym.make(args.game))
	obs = env.reset()
	obs, reward, done, info = env.step(1)

	# collect data
	tr_episodes, val_episodes,\
	tr_labels, val_labels,\
	test_episodes, test_labels = get_episodes(env_name=args.game, 
										steps=args.collection_steps, 
										collect_mode=args.agent_collect_mode)

	# encoder setup
	encoder = None
	if args.encoder == 'rand_cnn':
		encoder = RandCNN()
	
	assert encoder is not None

	# probe training, validation, testing
	state_vars_for_game = game_mappings[args.game]
	probe_handler = ProbeHandler(len(state_vars_for_game), encoder, state_vars_for_game, is_supervised=args.supervised)
	probe_handler.train(tr_episodes, tr_labels, epochs=int(args.epochs))
	probe_handler.validate(val_episodes, val_labels)
	probe_handler.test(test_episodes, test_labels)

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--game', default='Pong-v0', 
		help='atari game to use')
	parser.add_argument('--supervised', default=False,
		help='flag for fully supervised learning')
	parser.add_argument('--encoder', default='rand_cnn', 
		help='flag for the encoder method. possible options: rand_cnn, ...')
	parser.add_argument('--collection_steps', default=5000, 
		help='number of steps to collect episodes for')
	parser.add_argument('--agent_collect_mode', default='random_agent', 
		help='collection agent type')
	parser.add_argument('--epochs', default=100, 
		help='numbr of epochs to train for')

	return parser

if __name__ == "__main__":
	parser = parser()
	args = parser.parse_args()

	full_pipeline(args)