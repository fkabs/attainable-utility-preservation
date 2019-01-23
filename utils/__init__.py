#
# Some common utils
#
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


# %%
def policy_rollout(env, policy_f, state_f=lambda ts: ts.observation, reward_f=lambda ts: ts.reward):
    """
    Get sequences of states, actions and rewards from given env following given policy.
    """
    acts = []
    rewards = []
    ts = env.reset()
    sts = [state_f(ts)]
    while not ts.last():
        act = policy_f(ts)
        ts = env.step(act)
        acts.append(act)
        rewards.append(reward_f(ts))
        sts.append(state_f(ts))
    return sts, acts, rewards


# %% for vizualizations:
def get_frame(step, x=0, y=-1):
    color_state = step.observation['RGB']
    return np.moveaxis(color_state, x, y)


def gen_images(acts, env):
    ts = env.reset()
    return [get_frame(ts)] + [get_frame(env.step(act)) for act in acts]


def plot_images_to_ani(images, interval=250, blit=True, repeat_delay=1000):
    """
    Usage:
        from IPython.display import HTML
        from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game

        acts = ... action sequence for some policy ...

        HTML(plot_images_to_ani(gen_images(acts, sokoban_game(level=0))).to_jshtml())
    """
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    ims = [[plt.imshow(im, animated=True)] for im in images]
    ani = animation.ArtistAnimation(plt.gcf(), ims, interval=interval, blit=blit, repeat_delay=repeat_delay)
    return ani
