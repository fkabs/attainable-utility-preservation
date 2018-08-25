from __future__ import print_function
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from agents.dqn import DQNAgent
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_ball_bot as ball, side_effects_spontaneous_combustion as fire, side_effects_sushi_bot as sushi,\
    side_effects_vase as vase
from collections import namedtuple

MAX_LEN = 40


def derive_possible_rewards(env_class, kwargs):
    """
    Derive a subset of possible reward functions for the given environment.

    :param env_class:
    :param kwargs:
    :return:
    """
    env = env_class(**kwargs)
    time_step = env.reset()

    functions = []
    # First, all of the positions the agent might want to reach - inaccessible position Q-functions shouldn't change
    free_spaces = np.where(time_step.observation['board'] == env._value_mapping[' '])
    for space in zip(free_spaces[0], free_spaces[1]):
        fn = lambda obs: int(obs['board'][space] == env.AGENT_CHR) * env.GOAL_REWARD - env.MOVEMENT_REWARD
        fn.name = str(space)
        functions.append(fn)

    """
    if env_class == sokoban.SideEffectsSokobanEnvironment and kwargs.get('level') == 1:
        special = [('coin', -1)]
    else:
        special = []
    """

    return functions


def run_episode(agent, env, epsilon=None, save_frames=False, render_ax=None):
    """
    Run the episode with given greediness, recording and saving the frames if desired.
    """
    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))
        if render_ax:
            render_ax.imshow(np.moveaxis(time_step.observation['RGB'], 0, -1), animated=True)
            #plt.pause(0.001)

    ret = 0  # cumulative return
    frames = []

    time_step = env.reset()
    handle_frame(time_step)
    #print("\n" + str(agent.get_q(time_step.observation)))

    for t in itertools.count():
        action = agent.act(time_step.observation, eps=epsilon)
        time_step = env.step(action)
        handle_frame(time_step)
        loss = agent.learn(time_step, action)

        ret += time_step.reward
        if time_step.last() or t >= MAX_LEN:
            break

    return ret, t, env._calculate_episode_performance(time_step), loss, frames


def generate_run_agents(env_class, kwargs, num_episodes, render_ax, score_ax):
    """
    Generate one normal agent and a subset of possible rewards for the environment.

    :param env_class: class object, expanded with random reward-generation methods.
    :param kwargs: environmental intialization parameters.
    :param num_episodes:
    :param render_ax: PyPlot axis on which rendering can take place.
    :param score_ax: PyPlot axis on which scores can be plotted.
    """
    agents, movies = [], []
    penalty_functions = derive_possible_rewards(env_class, kwargs)

    EpisodeStats = namedtuple("EpisodeStats", ["lengths", "rewards", "performance"])
    stats_dims = (2, num_episodes)  # DQN vs AUP
    stats = EpisodeStats(lengths=np.zeros(stats_dims), rewards=np.zeros(stats_dims),
                         performance=np.zeros(stats_dims))

    env = env_class(**kwargs)
    actions_num, world_shape = env.action_spec().maximum + 1, env.observation_spec()['board'].shape

    global_step = tf.train.get_or_create_global_step()
    with tf.Session() as sess:
        agent = DQNAgent(sess, world_shape, actions_num, env, frames_state=2,
                         experiment_dir=os.path.join('side_grids_camp', 'experiments', env.name, name),
                         replay_memory_size=1500, replay_memory_init_size=500,
                         update_target_estimator_every=300, discount_factor=.99,
                         epsilon_start=1.0, epsilon_end=0.2, epsilon_decay_steps=50000, batch_size=512,
                               restore=False)
        agents.append(agent)
        agents[-1].name = name
        render_ax.set_xlabel("DQN")
        print("Training {} agent.".format(agent.name))

        # Begin training
        for i_episode in range(num_episodes):
            stats.rewards[i_agent, i_episode], stats.lengths[i_agent, i_episode], \
            stats.performance[i_agent, i_episode], loss, _ = run_episode(agent, env)
            if i_episode % 500 == 0 and i_episode:
                ret, _, _, _, _ = run_episode(agent, env, epsilon=0)
                print("Greedy ret: {}".format(ret))
            print("\rEpisode {}/{}, loss: {}, reward: {}, length: {}".format(i_episode + 1, num_episodes, loss,
                                                                             stats.rewards[i_agent, i_episode], stats.lengths[i_agent, i_episode]), end="")
        print("\n")

        agent.save()
        score_ax.plot(range(num_episodes), stats.rewards[i_agent])  # plot performance

        final_ret, _, _, _, frames = run_episode(agent, env, epsilon=0, save_frames=True)  # get frames from final policy
        movies.append((agent.name, frames))
        print("Greedy score: {}".format(final_ret))

    return agents, stats, movies
