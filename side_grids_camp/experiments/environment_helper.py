from __future__ import print_function
import itertools
import matplotlib.pyplot as plt
import numpy as np
from agents.aup import AUPAgent
from ai_safety_gridworlds.environments.shared import safety_game


def derive_possible_rewards(env):
    """
    Derive possible reward functions for the given environment.

    :param env: Environment.
    """
    def state_lambda(original_board_str):
        return lambda obs: int(obs == original_board_str) * env.GOAL_REWARD
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


def run_episode(agent, env, save_frames=False, render_ax=None):
    """
    Run the episode, recording and saving the frames if desired.

    :param save_frames: Whether to save frames from the final performance.
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
        if env.name == 'survival':
            actions.append(safety_game.Actions.NOTHING)  # disappearing frame
        max_len = len(actions)

    for i in itertools.count():
        if time_step.last() or (hasattr(agent, 'get_actions') and i >= max_len):
            break
        action = actions[i] if hasattr(agent, 'get_actions') else agent.act(time_step.observation)
        time_step = env.step(action)
        handle_frame(time_step)

    return float(env.episode_return), max_len, float(env._episodic_performances[-1]), frames
