import gym
from ram_dict import ram_state_dict


class AtariRAMWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env_name = self.env.spec.id

        # check that the game name is in the dictionary
        game_name = get_game_dict_key(self.env_name)
        assert game_name.lower(
        ) in ram_state_dict, f"{game_name} is not in RAM dictionary"

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, self.info(info)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def info(self, info):
        ram = self.env.unwrapped.ale.getRAM()
        label_dict = ram2label(self.env_name, ram)
        info["labels"] = label_dict
        return info


def get_game_dict_key(game_name):
    """Gets game name from word before first dash"""
    return game_name.split("-")[0]


def ram2label(env_name, ram):
    game_name = get_game_dict_key(env_name)
    if game_name.lower() in ram_state_dict:
        label_dict = {k: ram[ind]
                      for k, ind in ram_state_dict[game_name.lower()].items()}
    else:
        assert False, f"{game_name} is not in RAM dictionary"
    return label_dict
