from __future__ import print_function
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from agents.aup import AUPAgent


def derive_possible_rewards(env):
    """
    Derive possible reward functions for the given environment.

    :param env: Environment.
    """
    def state_lambda(original_board_str):
        return lambda obs: int(str(obs['board']) == original_board_str) * env.GOAL_REWARD
    def explore(env, so_far=[]):
        board_str = str(env._last_observations['board'])
        if board_str not in states:
            states.add(board_str)
            fn = state_lambda(board_str)
            fn.state = board_str
            functions.append(fn)
            if not env._game_over:
                for action in range(env.action_spec().maximum + 1):
                    env.step(action)
                    explore(env, so_far + [action])
                    AUPAgent.restart(env, so_far)

    env.reset()
    states, functions = set(), []
    explore(env)
    env.reset()
    return functions


def run_episode(agent, env, save_frames=False, render_ax=None, save_dir=None):
    """
    Run the episode with given greediness, recording and saving the frames if desired.

    :param save_frames: Whether to save frames from the final performance.
    :param save_dir: Where to save memoized AUP data.
    """
    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))
        if render_ax:
            render_ax.imshow(np.moveaxis(time_step.observation['RGB'], 0, -1), animated=True)
            plt.pause(0.001)

    max_len = 8
    frames = []

    time_step = env.reset()
    handle_frame(time_step)
    if hasattr(agent, 'get_actions'):
        actions, _ = agent.get_actions(env, steps_left=max_len)
        max_len = len(actions)
    for i in itertools.count():  # TODO fix logic
        if time_step.last() or (hasattr(agent, 'get_actions') and i >= max_len):
            break
        action = actions[i] if hasattr(agent, 'get_actions') else agent.act(time_step.observation)
        time_step = env.step(action)
        handle_frame(time_step)

    # Save memoized data
    if save_dir and hasattr(agent, 'dir'):
        if not os.path.exists(agent.dir):
            os.makedirs(agent.dir)
        with open(os.path.join(agent.dir, "attainable.pkl"), 'wb') as a, \
                open(os.path.join(agent.dir, "cached.pkl"), 'wb') as c:
            pickle.dump(agent.attainable, a, pickle.HIGHEST_PROTOCOL)
            pickle.dump(agent.cached_actions, c, pickle.HIGHEST_PROTOCOL)

    return env.episode_return, max_len, env._episodic_performances[-1], frames
