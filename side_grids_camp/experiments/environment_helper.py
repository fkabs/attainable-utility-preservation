from __future__ import print_function
import itertools
import numpy as np
import matplotlib.pyplot as plt
from agents.aup import AUPAgent
from agents.dqn import DQNAgent
from collections import namedtuple




def derive_possible_rewards(env_class, kwargs):
    """
    Derive a subset of possible reward functions for the given environment.

    :param env_class: Environment constructor.
    :param kwargs: Configuration parameters.
    """
    def state_lambda(original_board_str, agent_value, empty_value):
        return lambda obs: int(str(obs['board']).replace(agent_value, empty_value)
                               == original_board_str) * env.GOAL_REWARD + env.MOVEMENT_REWARD

    env = env_class(**kwargs)
    env.reset()
    agent_value, empty_value = str(env._value_mapping[env.AGENT_CHR])[:-2], str(env._value_mapping[' '])[:-2]

    states, functions = set(), []
    # Randomly generate states
    for i in range(100):
        env.reset()
        for depth in range(10):
            time_step = env.step(np.random.choice(range(env.action_spec().maximum)))  # don't try null action

            # Remove agent from state
            board_str = str(time_step.observation['board']).replace(agent_value, empty_value)
            if board_str not in states and not time_step.last():
                states.add(board_str)
                fn = state_lambda(board_str, agent_value, empty_value)
                fn.state = board_str
                functions.append(fn)

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

    ret, frames = 0, []  # cumulative return

    time_step = env.reset()
    handle_frame(time_step)

    actions, _ = agent.get_actions(env, steps_left=12)
    for action in actions:
        time_step = env.step(action)
        handle_frame(time_step)

        ret += time_step.reward
        if time_step.last():
            break

    return ret, len(actions), env._calculate_episode_performance(time_step), frames


def generate_run_agents(env_class, kwargs, score_ax=None, render_ax=None):
    """
    Generate one normal agent and a subset of possible rewards for the environment.

    :param env_class: class object, expanded with random reward-generation methods.
    :param kwargs: environmental intialization parameters.
    :param num_episodes:
    :param score_ax: PyPlot axis on which scores can be plotted.
    :param render_ax: PyPlot axis on which rendering can take place.
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
        movies.append(('Vanilla' if i_agent == 0 else 'AUP', frames))

    #if score_ax:
    #    score_ax.plot(range(num_episodes), stats.rewards[i_agent])  # plot performance

    return stats, movies
