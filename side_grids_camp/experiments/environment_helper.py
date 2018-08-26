from __future__ import print_function
import itertools
import numpy as np
import matplotlib.pyplot as plt
from agents.aup import AUPAgent
from agents.dqn import DQNAgent
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_ball_bot as ball, side_effects_spontaneous_combustion as fire, side_effects_sushi_bot as sushi,\
    side_effects_vase as vase
from collections import namedtuple


def derive_possible_rewards(env_class, kwargs):
    """
    Derive a subset of possible reward functions for the given environment.

    :param env_class:
    :param kwargs:
    :return:
    """
    def build_lambda(space):
        return lambda obs: int(obs['board'][space] == env._value_mapping[env.AGENT_CHR]) * env.GOAL_REWARD + env.MOVEMENT_REWARD

    env = env_class(**kwargs)
    time_step = env.reset()

    functions = []
    # First, all of the positions the agent might want to reach - inaccessible position Q-functions shouldn't change
    free_spaces = np.where(time_step.observation['board'] == env._value_mapping[' '])
    for space in zip(free_spaces[0], free_spaces[1]):
        fn = build_lambda(space)
        fn.name = str(space)
        functions.append(fn)  # TODO test

    # TODO add special rewards

    return functions


def run_episode(agent, env, save_frames=False, render_ax=None):
    """
    Run the episode with given greediness, recording and saving the frames if desired.
    """
    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))
        if render_ax:
            render_ax.imshow(np.moveaxis(time_step.observation['RGB'], 0, -1), animated=True)
            plt.pause(0.001)

    ret = 0  # cumulative return
    frames = []

    time_step = env.reset()
    handle_frame(time_step)
    #print("\n" + str(agent.get_q(time_step.observation)))

    actions = []
    for t in itertools.count():
        actions.append(agent.act(env, actions))
        time_step = env.step(actions[-1])
        handle_frame(time_step)

        ret += time_step.reward
        if time_step.last():
            break

    return ret, t, env._calculate_episode_performance(time_step), frames


def generate_run_agents(env_class, kwargs, score_ax, render_ax):
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
    stats_dims = (2)  # DQN vs AUP
    stats = EpisodeStats(lengths=np.zeros(stats_dims), rewards=np.zeros(stats_dims),
                         performance=np.zeros(stats_dims))

    env = env_class(**kwargs)
    agents = [AUPAgent(), AUPAgent(penalty_functions)]
    for i_agent, agent in enumerate(agents):
        _, _, _, frames = run_episode(agent, env, save_frames=True, render_ax=render_ax)
        movies.append(('Normal' if i_agent == 0 else 'AUP', frames))

    #score_ax.plot(range(num_episodes), stats.rewards[i_agent])  # plot performance

    return stats, movies
