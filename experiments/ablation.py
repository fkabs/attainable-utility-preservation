from __future__ import print_function

# import matplotlib as mpl
# mpl.use('Agg')

import os
import warnings
import datetime
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment_helper import *
from ai_safety_gridworlds.environments import *
from agents.model_free_aup import ModelFreeAUPAgent


def plot_images_to_ani(framesets):
    """
    Animates all agent executions and returns the animation object.

    :param framesets: [("agent_name", frames),...]
    """
    if len(framesets) == 7:
        axs = [plt.subplot(3, 3, 2),
               plt.subplot(3, 3, 4), plt.subplot(3, 3, 5), plt.subplot(3, 3, 6),
               plt.subplot(3, 3, 7), plt.subplot(3, 3, 8), plt.subplot(3, 3, 9)]
    else:
        _, axs = plt.subplots(1, len(framesets), figsize=(5, 5 * len(framesets)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

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
    return animation.ArtistAnimation(plt.gcf(), ims, interval=400, blit=True, repeat_delay=200)


def run_game(game):
    env_class, env_kwargs = game
    render_fig, render_ax = plt.subplots(1, 1)
    render_fig.set_tight_layout(True)
    render_ax.get_xaxis().set_ticks([])
    render_ax.get_yaxis().set_ticks([])
    env_class.variant_name = env_class.name + '-' + str(env_kwargs['level'] if 'level' in env_kwargs else env_kwargs['variant'])
    print(env_class.variant_name)

    start_time = datetime.datetime.now()
    movies = run_agents(env_class, env_kwargs, render_ax=render_ax)

    # Save first frame of level for display in paper
    # render_ax.imshow(movies[0][1][0])
    render_fig.savefig(os.path.join(os.path.dirname(__file__), 'level_imgs', env_class.variant_name + '.pdf'),
                       bbox_inches='tight', dpi=350)
    plt.close(render_fig.number)

    print("Training finished; {} elapsed.\n".format(datetime.datetime.now() - start_time))
    ani = plot_images_to_ani(movies)
    ani.save(os.path.join(os.path.dirname(__file__), 'gifs', env_class.variant_name + '.gif'),
             writer='imagemagick', dpi=350)
    # plt.show()


def run_agents(env_class, env_kwargs, render_ax=None):
    """
    Generate and run agent variants.

    :param env_class: class object.
    :param env_kwargs: environmental intialization parameters.
    :param render_ax: PyPlot axis on which rendering can take place.
    """
    # Instantiate environment and agents
    env = env_class(**env_kwargs)
    model_free = ModelFreeAUPAgent(env, trials=1)
    state = (ModelFreeAUPAgent(env, state_attainable=True, trials=1))
    movies, agents = [], [ModelFreeAUPAgent(env, num_rewards=0, trials=1),  # vanilla (standard q-learner)
                          AUPAgent(attainable_Q=model_free.attainable_Q, baseline='start'),  # starting state
                          AUPAgent(attainable_Q=model_free.attainable_Q, baseline='inaction'),  # incation
                          AUPAgent(attainable_Q=model_free.attainable_Q, deviation='decrease'),  # decrease
                          AUPAgent(attainable_Q=state.attainable_Q, baseline='inaction', deviation='decrease'),  # relative reachability
                          model_free,  # model-free aup
                          AUPAgent(attainable_Q=model_free.attainable_Q)  # full AUP
                          ]

    for agent in agents:
        ret, _, perf, frames = run_episode(agent, env, save_frames=True, render_ax=render_ax)
        movies.append((agent.name, frames))
        print(agent.name, perf)

    return movies


if __name__ == '__main__':
    # set number of usable CPU cores
    NUM_CORES = mp.cpu_count()
    
    # set eaup variant
    eaup = None
    
    # no no-op action for eaup variants
    if eaup != None:
        safety_game.AGENT_LAST_ACTION = 3
    
    games = [
        (conveyor.ConveyorEnvironment, {'variant': 'vase'}),
        (conveyor.ConveyorEnvironment, {'variant': 'sushi'}),
        (burning.BurningEnvironment, {'level': 0}),
        (burning.BurningEnvironment, {'level': 1}),
        (box.BoxEnvironment, {'level': 0}),
        (sushi.SushiEnvironment, {'level': 0}),
        (vase.VaseEnvironment, {'level': 0}),
        (dog.DogEnvironment, {'level': 0}),
        (survival.SurvivalEnvironment, {'level': 0})
    ]

    # Plot setup
    plt.style.use('ggplot')
    
    # distribute experiments on all core
    # pool = mp.Pool(NUM_CORES)
    # results = pool.map(run_game, games)
    run_game(games[0])
