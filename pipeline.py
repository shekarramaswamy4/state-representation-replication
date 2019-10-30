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
	# train probe 
	probe_handler = ProbeHandler(3) # update number of state variables based on environment

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--game', default='Pong-v0', 
		help='atari game to use')


if __name__ == "__main__":
	parser = ()
	args = parser.parse_args()

	full_pipeline(args)