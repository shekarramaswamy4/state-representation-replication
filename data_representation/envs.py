import cv2
import gym
from gym import spaces
import time
import os
import numpy as np
from baselines import bench
from baselines.common.atari_wrappers import make_atari, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, \
    ClipRewardEnv, FrameStack
from a2c_ppo_acktr.envs import TransposeImage
from .wrapper import AtariRAMWrapper


def make_atari_env(env_name, seed=42, log_dir="./.atari_env_log_dir", run_folder_name="default_run"):
    """
    Used their value for seed=42.
    """

    # Baselines wrapper on Atari environments
    env = paper_preprocessing(env_name)
    env = AtariRAMWrapper(env)

    env.seed(seed)

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        env = bench.Monitor(
            env,
            os.path.join(log_dir, run_folder_name),
            allow_early_resets=False)

    obs_shape = env.observation_space.shape
    # just to have the color dimension first
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env, op=[2, 0, 1])

    # TODO: come back and look at if we want to have the
    # parallelized processing using VecPyTorch etc.
    return env


def paper_preprocessing(env_name):
    # no downsampling (image size remains (210, 160))
    # no framestacking

    # this takes care of:
    # action repetitions = 4,
    # max pooling = 2,
    # and No-Op action reset
    env = make_atari(env_name)
    # grayscaling = True
    env = GrayscaleWrapper(env)
    # end of life episode
    env = EpisodicLifeEnv(env)
    return env


class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(
                                                self.observation_space.shape[0], self.observation_space.shape[1], 1),
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame
