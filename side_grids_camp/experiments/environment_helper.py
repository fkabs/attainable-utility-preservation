import itertools
import numpy as np
import os
from agents.dqn import DQNAgent
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_ball_bot as ball, side_effects_spontaneous_combustion as fire, side_effects_sushi_bot as sushi,\
    side_effects_vase as vase


def generate_agents_environments(sess, env_class, kwargs, num_random=0):
    """
    Generate one normal agent and the given number of agents with randomly-seeded reward functions.

    :param sess: TensorFlow session.
    :param env_class: class object, expanded with random reward-generation methods.
    :param kwargs: environmental intialization parameters.
    :param num_random: how many reward functions to randomly seed.
    """
    num_agents = 1 + num_random
    agents, envs = [], []
    free_spaces, special = derive_possible_rewards(env_class, kwargs)
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
    free_spaces = np.array(zip(free_spaces[0], free_spaces[1]))

    if env_class == sokoban.SideEffectsSokobanEnvironment and kwargs.get('level') == 1:
        special = {'coin_multi': [1, -1]}
    else:
        special = {}

    return free_spaces, special


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
        agent.learn(time_step, action)

        ret += time_step.reward
        if time_step.last():
            break
    return ret, t, env._calculate_episode_performance(time_step), frames
