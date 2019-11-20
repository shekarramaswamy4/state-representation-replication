import time
from .envs import make_atari_env
from .label_preprocess import remove_duplicates, remove_low_entropy_labels
import numpy as np
from collections import deque
from itertools import chain
import torch


def get_random_episodes(env_name="Pong-v0",
                        steps=50000,
                        seed=42,
                        train_mode="probe",
                        min_episode_length=64,
                        entropy_threshold=0.6):

    env = make_atari_env(env_name, seed=seed)
    env.reset()

    print('-------Collecting samples----------')
    # (n_processes * n_episodes * episode_len)
    episodes = [[]]
    episode_labels = [[]]
    for step in range(steps):
        # Take action using a random policy
        action = torch.tensor(
            np.array(np.random.randint(
                1, env.action_space.n)))

        obs, reward, done, info = env.step(action)

        obs = torch.tensor(obs)

        if not done:
            episodes[-1].append(obs.clone().float())
            if "labels" in info.keys():
                episode_labels[-1].append(info["labels"])
        else:
            episodes.append([obs.clone().float()])
            if "labels" in info.keys():
                episode_labels.append([info["labels"]])

            # reset here because in a vectorized environment, the environment
            # gets automatically reset when each environment is done
            env.reset()

    env.close()

    ep_inds = [i for i in range(len(episodes)) if len(
        episodes[i]) > min_episode_length]
    episodes = [episodes[i] for i in ep_inds]
    episode_labels = [episode_labels[i] for i in ep_inds]
    episode_labels, entropy_dict = remove_low_entropy_labels(
        episode_labels, entropy_threshold=entropy_threshold)

    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(inds)

    if train_mode == "train_encoder":
        assert len(
            inds) > 1, "Not enough episodes to split into train and val. You must specify enough steps to get at least two episodes"
        split_ind = int(0.8 * len(inds))
        tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]
        return tr_eps, val_eps

    if train_mode == "probe":
        val_split_ind, te_split_ind = int(
            0.7 * len(inds)), int(0.8 * len(inds))
        assert val_split_ind > 0 and te_split_ind > val_split_ind,\
            "Not enough episodes to split into train, val and test. You must specify more steps"
        tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[
            te_split_ind:]
        tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], \
            episode_labels[val_split_ind:te_split_ind], episode_labels[te_split_ind:]
        test_eps, test_labels = remove_duplicates(
            tr_eps, val_eps, test_eps, test_labels)
        test_ep_inds = [i for i in range(
            len(test_eps)) if len(test_eps[i]) > 1]
        test_eps = [test_eps[i] for i in test_ep_inds]
        test_labels = [test_labels[i] for i in test_ep_inds]
        return tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels

    if train_mode == "dry_run":
        return episodes, episode_labels


if __name__ == "__main__":
    tr_episodes, val_episodes,\
        tr_labels, val_labels,\
        test_episodes, test_labels = get_random_episodes(env_name="Breakout-v0",
                                                         steps=5000,
                                                         min_episode_length=2)
    print(len(tr_episodes))
    print(len(val_episodes))
    print(len(tr_labels))
    print(len(val_labels))
    print(len(test_episodes))
    print(len(test_labels))

# for i in range(1000):
#     env.render()
#     obs, reward, done, info = env.step(
#         env.action_space.sample())  # take a random action
#     print(info)
#     print(env.unwrapped.ale.getRAM())
#     break
#     if done:
#         print(i)
#         break
#     # time.sleep(1)
# env.close()
