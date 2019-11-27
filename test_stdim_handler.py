import argparse
import os
import pickle

from handlers.stdim_handler import StDimHandler
from encoders.rand_cnn import RandCNN
from data_representation.get_data import get_random_episodes

def test_stdim(args, gen_new_data = False):
    # ordering for state variables in dictionary form
    game_mappings = {'Pong-v0': {'player_y': 0, 'enemy_y': 1, 'ball_x': 2, 'ball_y': 3, 'enemy_score': 4, 'player_score': 5}}
    collection_lengths = {'Pong-v0' : 2}

    data_dir=f"{args.game}-{args.collection_steps}"
    
    if gen_new_data:
        os.makedirs(data_dir, exist_ok=True)
        print(f"--- Writing new data to {data_dir} ---")
        # collect new data
        tr_episodes, val_episodes,\
        tr_labels, val_labels,\
        test_episodes, test_labels = get_random_episodes(env_name=args.game, 
                                            steps=args.collection_steps, 
                                            min_episode_length=collection_lengths[args.game])
        pickle.dump(tr_episodes,   open(f"{data_dir}/tr_episodes.test", "wb"))
        pickle.dump(tr_labels,     open(f"{data_dir}/tr_labels.test", "wb"))
        pickle.dump(val_episodes,  open(f"{data_dir}/val_episodes.test", "wb"))
        pickle.dump(val_labels,    open(f"{data_dir}/val_labels.test", "wb"))
        pickle.dump(test_episodes, open(f"{data_dir}/test_episodes.test", "wb"))
        pickle.dump(test_labels,   open(f"{data_dir}/test_labels.test", "wb"))
    else:
        print(f"--- Loading existing data from {data_dir} ---")
        tr_episodes = pickle.load(    open(f"{data_dir}/tr_episodes.test", "rb"))
        tr_labels = pickle.load(        open(f"{data_dir}/tr_labels.test", "rb"))
        val_episodes = pickle.load(  open(f"{data_dir}/val_episodes.test", "rb"))
        val_labels = pickle.load(      open(f"{data_dir}/val_labels.test", "rb"))
        # test_episodes = pickle.load(open(f"{data_dir}/test_episodes.test", "rb"))
        # test_labels = pickle.load(    open(f"{data_dir}/test_labels.test", "rb"))

    handler = StDimHandler()
    handler.train(tr_episodes, tr_labels, val_episodes=val_episodes, val_labels=val_labels)
    
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
    test_stdim(args,gen_new_data=False)