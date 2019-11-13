
def testing():
    pass
    # observation = env.reset()
    # print(observation.shape)
    # print(env.observation_space)
    # print(env.action_space)
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
# from atariari.benchmark.episodes import get_episodes
# tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels = get_episodes(env_name="PitfallNoFrameskip-v4", steps=50000, collect_mode="random_agent")
