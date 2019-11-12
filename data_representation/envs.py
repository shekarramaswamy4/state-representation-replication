import gym
import time
import os
from baselines import bench
from baselines.common.atari_wrappers import make_atari, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, \
    ClipRewardEnv, FrameStack
from .wrapper import AtariRAMWrapper


def make_atari_env(env_name, seed=42, log_dir="./log_dir", run_folder_name="default_run"):
    """
    Used their value for seed=42.
    """

    # Baselines wrapper on Atari environments
    env = AtariRAMWrapper(env)
    env = paper_preprocessing(env)
    env.seed(seed)

    if log_dir is not None:
        env = bench.Monitor(
            env,
            os.path.join(log_dir, run_folder_name),
            allow_early_resets=False)

    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env, op=[2, 0, 1])

    return env


def paper_preprocessing(env):
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
