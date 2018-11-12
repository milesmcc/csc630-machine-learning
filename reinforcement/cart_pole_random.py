import gym

env = gym.make("CartPole-v0")
obs = env.reset()
for _ in range(5000):
    env.render()
    print(obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)