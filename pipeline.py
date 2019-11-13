# imports
import argparse
import gym
import time

from probe.probe_handler import ProbeHandler
from encoders.rand_cnn import RandCNN 
from atariari.benchmark.wrapper import AtariARIWrapper
from atariari.benchmark.episodes import get_episodes

'''
run pipeline.py to run the experiment end to end:
collect data
train encoder
run probe and get results
'''

def full_pipeline(args):
	print('running full pipeline')
	# get training, validation, testing data
	env = AtariARIWrapper(gym.make('PongNoFrameskip-v4'))
	obs = env.reset()
	obs, reward, done, info = env.step(1)


	tr_episodes, val_episodes,\
	tr_labels, val_labels,\
	test_episodes, test_labels = get_episodes(env_name="Pong-v0", 
										steps=5000, 
										collect_mode="random_agent")
	# train encoder here if it's one that needs training (ex. dim, not randcnn)

	encoder = RandCNN()
	# probe handler needs to know how many state variables we are using
	probe_handler = ProbeHandler(3, encoder, is_supervised=args.supervised)
	probe_handler.train(tr_episodes, tr_labels)
	probe_handler.validate(val_episodes, val_labels)
	probe_handler.test(test_episodes, test_labels)

	# evaluate frozen encoder + trained probes w/ f1 score (setup in probe handler)

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--game', default='Pong-v0', 
		help='atari game to use')
	parser.add_argument('--supervised', default=False,
		help='flag for fully supervised learning')
	parse.add_argument('--encoder', default='rand_cnn', 
		help='flag for the encoder method. possible options: rand_cnn, ')

if __name__ == "__main__":
	parser = ()
	args = parser.parse_args()

	full_pipeline(args)