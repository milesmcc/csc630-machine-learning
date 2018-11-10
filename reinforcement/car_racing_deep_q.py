# From https://tensorforce.readthedocs.io/en/latest/

import numpy as np

from tensorforce.agents import TRPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

env = OpenAIGym('CarRacing-v0', visualize=True)

network_spec = [
    dict(type='flatten'),
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

agent = TRPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec
)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    if r.episode > 400:
        env.visualize = True
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=None, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)