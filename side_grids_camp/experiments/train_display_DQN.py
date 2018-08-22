from __future__ import print_function
from collections import namedtuple
import itertools
import numpy as np
import sys
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from agents.dqn import StateProcessor, Estimator, DQNAgent
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game


def generate_agents_environments(sess, env_class, kwargs, num_random=0):
    """
    Generate one normal agent and the given number of agents with randomly-seeded reward functions.

    :param sess: TensorFlow session.
    :param env_class: class object, expanded with random reward-generation methods.
    :param kwargs: environmental intialization parameters.
    :param num_random: how many reward functions to randomly seed.
    :return:
    """

    num_agents = 1 + num_random
    agents, envs = [], []

    for i in range(num_agents):
        envs.append(env_class(custom_goal=(4, 4), **kwargs))
        actions_num, world_shape = envs[-1].action_spec().maximum + 1, envs[-1].observation_spec()['board'].shape

        #graph = tf.Graph()
        #with graph.as_default():
        agents.append(DQNAgent(sess, world_shape, actions_num, envs[-1], frames_state=2,
                         experiment_dir=os.path.join('side_grids_camp', 'experiments',
                                                     envs[-1].name, str(i)) if i==0 else None,
                         replay_memory_size=10000, replay_memory_init_size=500,
                         update_target_estimator_every=250, discount_factor=1.0,
                         epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, batch_size=32))

    return agents, envs


def run_episode(agent, env, epsilon=None, save_frames=False):
    """
    Run the episode with given greediness, recording and saving the frames if desired.
    """
    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))

    ret = 0  # cumulative return
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
    return ret, t, env._calculate_episode_performance(time_step), frames


def plot_images_to_ani(framesets):
    """
    Animates all agent executions and returns the animation object.

    :param framesets: [("agent_name", frames),...]
    """
    fig, axs = plt.subplots(1, len(framesets), figsize=(5, 5 * len(framesets)))

    max_runtime = max([len(frames) for _, frames in framesets])

    ims, zipped = [], zip(framesets, axs if len(framesets) > 1 else [axs])  # handle 1-agent case
    for i in range(max_runtime):
        ims.append([])
        for (agent_name, frames), ax in zipped:
            if i == 0:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(agent_name)
            ims[-1].append(ax.imshow(frames[min(i, len(frames) - 1)], animated=True))
    return animation.ArtistAnimation(plt.gcf(), ims, interval=250, blit=True, repeat_delay=0)


start_time = datetime.datetime.now()
global_step = tf.Variable(0, name="global_step", trainable=False)

game, kwargs = sokoban_game, {'level': 0}
scores = plt.figure()
plt.ylabel("Score")
plt.xlabel("Episode")

tf.reset_default_graph()
with tf.Session() as sess:
    agents, envs = generate_agents_environments(sess, game, kwargs, num_random=0)

    num_episodes = 200
    EpisodeStats = namedtuple("EpisodeStats", ["lengths", "rewards", "performance"])
    stats = EpisodeStats(lengths=np.zeros((len(agents), num_episodes)), rewards=np.zeros((len(agents), num_episodes)),
                         performance=np.zeros((len(agents), num_episodes)))

    movies = []
    for i_agent, (agent, env) in enumerate(zip(agents, envs)):
        for i_episode in range(num_episodes):
                stats.lengths[i_agent, i_episode], stats.rewards[i_agent, i_episode], \
                stats.performance[i_agent, i_episode], _ = run_episode(agent, env)
        agent.save()

        plt.plot(range(num_episodes), stats.rewards[i_agent])  # plot performance

        _, _, _, frames = run_episode(agent, env, epsilon=.1, save_frames=True)  # get frames from final policy
        movies.append((str(i_agent), frames))
    plt.show()  # show performance

    print("\nTraining finished for {}; {} elapsed.".format(game.name, datetime.datetime.now() - start_time))
    ani = plot_images_to_ani(movies)
    ani.save(os.path.join('side_grids_camp', 'gifs', sokoban_game.name + '.gif'), writer='imagemagick')
    plt.show()



