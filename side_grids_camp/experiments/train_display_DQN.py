from __future__ import print_function
import itertools
import numpy as np
import sys
import os
import tensorflow as tf
from collections import namedtuple
import datetime
from matplotlib import pyplot as plt

from agents.dqn import StateProcessor, Estimator, DQNAgent
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game

EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])
global_step = tf.Variable(0, name="global_step", trainable=False)

env = sokoban_game(level=0)
actions_num = env.action_spec().maximum + 1
world_shape = env.observation_spec()['board'].shape
e = Estimator(actions_num, world_shape[0], world_shape[1], scope="test")
sp = StateProcessor(world_shape[0], world_shape[1])
frames_state = 2
batch_size = 32

start_time = datetime.datetime.now()
num_episodes = 20
stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                     episode_rewards=np.zeros(num_episodes))
# TODO what are the network valuations? normalize?

import matplotlib.animation as animation


def plot_images_to_ani(framesets):
    """
    Animates all agent executions and returns the animation object.

    :param framesets: [("agent_name", frames),...]
    """
    fig, axs = plt.subplots(1, len(framesets), figsize=(5, 5 * len(framesets)))

    max_runtime = max([len(frames) for _, frames in framesets])

    ims, zipped = [], zip(framesets, axs)
    for i in range(max_runtime):
        ims.append([])
        for (agent_name, frames), ax in zipped:
            if i == 0:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(agent_name)
            ims[-1].append(ax.imshow(frames[min(i, len(frames) - 1)], animated=True))
    return animation.ArtistAnimation(plt.gcf(), ims, interval=250, blit=True, repeat_delay=1000)


def run_episode(agent, env, epsilon=None, save_frames=False):
    """
    Run the episode with given greediness, recording and saving the frames if desired.
    """
    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))

    ret = 0
    frames = []
    time_step = env.reset()
    handle_frame(time_step)

    for t in itertools.count():
        action = agent.act(time_step.observation, eps=epsilon)
        time_step = env.step(action)
        handle_frame(time_step)
        loss = agent.learn(time_step, action)

        print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
            t, agent.total_t, i_episode + 1, num_episodes, loss), end="")
        sys.stdout.flush()

        ret += time_step.reward
        if time_step.last():
            break
    return ret, t, frames


tf.reset_default_graph()
with tf.Session() as sess:
    agent = DQNAgent(sess,
                     world_shape,
                     actions_num,
                     env,
                     frames_state=frames_state,
                     experiment_dir=None,  # os.path.join(env.name, "normal"),  # TODO implement
                     replay_memory_size=10000,  # 10000
                     replay_memory_init_size=500,  # 3000
                     update_target_estimator_every=250,  # 500
                     discount_factor=0.99,
                     epsilon_start=1.0,
                     epsilon_end=0.1,
                     epsilon_decay_steps=50000,
                     batch_size=batch_size)

    for i_episode in range(num_episodes):
        stats.episode_lengths[i_episode], stats.episode_rewards[i_episode], _ = run_episode(agent, env)

    agent.save()
    # Get frames from final policy
    _, _, frames = run_episode(agent, env, epsilon=0, save_frames=True)

print("\nTraining finished; {} elapsed.".format(datetime.datetime.now() - start_time))
ani = plot_images_to_ani([("DQN greedy", frames), ("DQN .8-greedy", frames)])
ani.save(os.path.join('side_grids_camp', 'gifs', env.name + '.gif'), writer='imagemagick')
plt.show()

plt.figure()
plt.plot(range(num_episodes), stats.episode_rewards)
plt.ylabel("Observed reward")
plt.xlabel("Episode")
#plt.show()
