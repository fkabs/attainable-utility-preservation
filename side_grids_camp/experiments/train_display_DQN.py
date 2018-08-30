from __future__ import print_function
from environment_helper import *
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_ball_bot as ball, side_effects_spontaneous_combustion as fire, side_effects_sushi_bot as sushi,\
    side_effects_vase as vase
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


start_time = datetime.datetime.now()
games = [sokoban.SideEffectsSokobanEnvironment, vase.SideEffectsVaseEnvironment]
game, kwargs = games[1], {'level': 0}

# Plot setup
plt.switch_backend('TkAgg')
plt.style.use('ggplot')

scores, score_ax = plt.subplots(1, 1)
plt.ylabel("Score")
plt.xlabel("Episode")

# Live rendering setup
render, render_ax = plt.subplots(1, 1)
render_ax.get_xaxis().set_ticks([])
render_ax.get_yaxis().set_ticks([])


stats, movies = generate_run_agents(game, kwargs, score_ax=score_ax, render_ax=render_ax)
plt.close(render.number)

#plt.show()  # show performance

print("Training finished for {}; {} elapsed.".format(game.name, datetime.datetime.now() - start_time))
ani = plot_images_to_ani(movies)
ani.save(os.path.join('side_grids_camp', 'gifs', game.name + '.gif'), writer='imagemagick', dpi=6000)
plt.show()
