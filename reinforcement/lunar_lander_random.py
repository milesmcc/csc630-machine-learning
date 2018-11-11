import gym

env = gym.make("LunarLander-v2")
obs = env.reset()
for _ in range(5000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
    print(obs, reward, done, info)