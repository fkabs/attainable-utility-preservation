from __future__ import print_function
from environment_helper import *
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_sushi_bot as sushi, side_effects_vase as vase, survival_incentive as survival, \
    side_effects_conveyor_belt as conveyor, side_effects_coffee_bot as coffee
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_images_to_ani(framesets):
    """
    Animates all agent executions and returns the animation object.

    :param framesets: [("agent_name", frames),...]
    """
    fig, axs = plt.subplots(1, len(framesets), figsize=(5, 5 * len(framesets)))
    fig.set_tight_layout(True)

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


def run_game(game, kwargs):
    render, render_ax = plt.subplots(1, 1)
    render.set_tight_layout(True)
    render_ax.get_xaxis().set_ticks([])
    render_ax.get_yaxis().set_ticks([])
    game.variant_name = game.name + '-' + str(kwargs['level'] if 'level' in kwargs else kwargs['variant'])

    start_time = datetime.datetime.now()
    stats, movies = generate_run_agents(game, kwargs, render_ax=render_ax)
    plt.close(render.number)

    print("Training finished for {}; {} elapsed.".format(game.name, datetime.datetime.now() - start_time))
    ani = plot_images_to_ani(movies)
    ani.save(os.path.join(os.path.dirname( __file__ ), game.variant_name, 'performance.gif'),
             writer='imagemagick', dpi=350)
    #plt.show()


games = [sokoban.SideEffectsSokobanEnvironment, sushi.SideEffectsSushiBotEnvironment,
         vase.SideEffectsVaseEnvironment, coffee.SideEffectsCoffeeBotEnvironment,
         survival.SurvivalIncentiveEnvironment]

# Plot setup
plt.switch_backend('TkAgg')
plt.style.use('ggplot')

# Levels for which we run multiple variants
#for var in ['vase', 'sushi']:
#    run_game(conveyor.ConveyorBeltEnvironment, {'variant': var})
for level in [0, 1]:
    run_game(burning.SideEffectsBurningBuildingEnvironment, {'level': level})

# The rest
for game in games:
    run_game(game, {'level': 0})
