import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from agents.dqn import DQNAgent
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_ball_bot as ball, side_effects_spontaneous_combustion as fire, side_effects_sushi_bot as sushi,\
    side_effects_vase as vase


def derive_possible_rewards(env_class, kwargs):
    """
    Derive a subset of possible reward functions for the given environment.

    :param env_class:
    :param kwargs:
    :return:
    """
    env = env_class(**kwargs)
    time_step = env.reset()

    # First, all of the positions the agent might want to reach - inaccessible position Q-functions shouldn't change
    free_spaces = np.where(time_step.observation['board'] == env._value_mapping[' '])
    free_spaces = zip(free_spaces[0], free_spaces[1])
    free_spaces = [(str(tup), tup) for tup in free_spaces]

    if env_class == sokoban.SideEffectsSokobanEnvironment and kwargs.get('level') == 1:
        special = [('coin', -1)]
    else:
        special = []

    return free_spaces, special


def generate_agents_environments(sess, env_class, kwargs):
    """
    Generate one normal agent and a subset of possible rewards for the environment.

    :param sess: TensorFlow session.
    :param env_class: class object, expanded with random reward-generation methods.
    :param kwargs: environmental intialization parameters.
    """
    agents, envs = [], []
    free_spaces, special = derive_possible_rewards(env_class, kwargs)

    for i, (name, reward_arg) in enumerate([('Vanilla', None)] + free_spaces + special):
        envs.append(env_class(custom_goal=reward_arg, **kwargs))
        actions_num, world_shape = envs[-1].action_spec().maximum + 1, envs[-1].observation_spec()['board'].shape
        with tf.variable_scope(str(i)):
            agents.append(DQNAgent(sess, world_shape, actions_num, envs[-1], frames_state=2,
                             experiment_dir=os.path.join('side_grids_camp', 'experiments', envs[-1].name, name),
                             replay_memory_size=10000, replay_memory_init_size=500,
                             update_target_estimator_every=250, discount_factor=1.0,
                             epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, batch_size=32,
                                   restore=False))
            agents[-1].name = name
    return agents, envs


def run_episode(agent, env, epsilon=None, save_frames=False, render=False):
    """
    Run the episode with given greediness, recording and saving the frames if desired.
    """
    def handle_frame(time_step):
        if save_frames:
            frames.append(np.moveaxis(time_step.observation['RGB'], 0, -1))
        if render:
            plt.imshow(np.moveaxis(time_step.observation['RGB'], 0, -1), animated=True)
            plt.pause(0.05)

    ret = 0  # cumulative return
    frames = []

    time_step = env.reset()
    if render:
        fig, ax = plt.subplots(1, 1)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel(agent.name)
        handle_frame(time_step)

    for t in itertools.count():
        action = agent.act(time_step.observation, eps=epsilon)
        time_step = env.step(action)
        handle_frame(time_step)
        loss = agent.learn(time_step, action)

        ret += time_step.reward
        if time_step.last():
            break
    if render:
        plt.close(fig)

    return ret, t, env._calculate_episode_performance(time_step), frames
