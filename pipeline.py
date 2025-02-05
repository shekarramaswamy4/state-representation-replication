import argparse
import gym
import time

from handlers.probe_handler import ProbeHandler
from handlers.cpc_handler import CPCHandler

from encoders.rand_cnn import RandCNN 

from data_representation.get_data import get_random_episodes

# ordering for state variables in dictionary form
game_mappings = {'Pong-v0': {'player_y': 0, 'enemy_y': 1, 'ball_x': 2, 'ball_y': 3, 'enemy_score': 4, 'player_score': 5}, 
				'Breakout': {'ball_x': 0, 'ball_y': 1, 'player_x': 2, 'blocks_hit_count': 3, 'score': 4, 'block_bit_map': 5}}
collection_lengths = {'Pong-v0' : 2, 'Breakout' : 2}

def full_pipeline(args):
	# collect data
	tr_episodes, val_episodes,\
	tr_labels, val_labels,\
	test_episodes, test_labels = get_random_episodes(env_name=args.game, 
										steps=args.collection_steps, 
										min_episode_length=collection_lengths[args.game])
	

	# encoder setup
	encoder = None
	if args.encoder == 'rand_cnn':
		encoder = RandCNN()
	elif args.encoder == 'cpc':
		input_shape = tr_episodes[0][0].shape
		encoder = CPCHandler(input_shape)
	elif args.encoder == 'stdim':
		print('implement stdim encoder in pipeline.py')
	
	assert encoder is not None

	# probe training, validation, testing
	state_vars_for_game = game_mappings[args.game]
	probe_handler = ProbeHandler(len(state_vars_for_game), encoder, state_vars_for_game, run_id=args.run_id, is_supervised=args.supervised)
	probe_handler.train(tr_episodes, tr_labels, val_episodes=val_episodes, val_labels=val_labels, epochs=int(args.epochs))
	# probe_handler.validate(val_episodes, val_labels)
	probe_handler.test(test_episodes, test_labels)

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--game', default='Pong-v0', 
		help='atari game to use')
	parser.add_argument('--supervised', default=False,
		help='flag for fully supervised learning')
	parser.add_argument('--encoder', default='rand_cnn', 
		help='flag for the encoder method. possible options: rand_cnn, ...')
	parser.add_argument('--collection_steps', default=50000, 
		help='number of steps to collect episodes for')
	parser.add_argument('--agent_collect_mode', default='random_agent', 
		help='collection agent type')
	parser.add_argument('--epochs', default=100, 
		help='numbr of epochs to train for')
	parser.add_argument('--run_id', default='', 
		help='save file identifier for runs')

	return parser

if __name__ == "__main__":
	parser = parser()
	args = parser.parse_args()

	full_pipeline(args)