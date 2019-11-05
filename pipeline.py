# imports
import argparse

from probe.probe_handler import ProbeHandler

'''
run pipeline.py to run the experiment end to end:
collect data
train encoder
run probe and get results
'''

def full_pipeline(args):
	print('running full pipeline')
	# get training, validation, testing data
	# train encoder

	# probe handler needs to know how many state variables we are using
	# if fully supervised, need an encoder as well.
	probe_handler = ProbeHandler(3, encoder=None, is_supervised=args.supervised)
	# probe_handler.train(DATA)

	# evaluate frozen encoder + trained probes

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