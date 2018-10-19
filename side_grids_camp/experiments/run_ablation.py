from __future__ import print_function
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_sushi_bot as sushi, side_effects_vase as vase, survival_incentive as survival, \
    side_effects_conveyor_belt as conveyor, side_effects_coffee_bot as coffee
from agents.aup_tab_q import AUPTabularAgent
from environment_helper import *
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    return animation.ArtistAnimation(plt.gcf(), ims, interval=250, blit=True, repeat_delay=0)


def run_game(game, kwargs):
    render_fig, render_ax = plt.subplots(1, 1)
    render_fig.set_tight_layout(True)
    render_ax.get_xaxis().set_ticks([])
    render_ax.get_yaxis().set_ticks([])
    game.variant_name = game.name + '-' + str(kwargs['level'] if 'level' in kwargs else kwargs['variant'])

    start_time = datetime.datetime.now()
    movies = run_agents(game, kwargs, render_ax=render_ax)
    render_ax.imshow(movies[0][1][0])
    render_fig.savefig(os.path.join(os.path.dirname(__file__), game.variant_name, game.name + '.eps'),
                       bbox_inches='tight', dpi=350)
    plt.close(render_fig.number)

    print("Training finished for {}; {} elapsed.".format(game.name, datetime.datetime.now() - start_time))
    ani = plot_images_to_ani(movies)
    ani.save(os.path.join(os.path.dirname(__file__), game.variant_name, 'perf.gif'),
             writer='imagemagick', dpi=350)
    plt.show()


def run_agents(env_class, env_kwargs, render_ax=None):
    """
    Generate and run agent variants.

    :param env_class: class object.
    :param env_kwargs: environmental intialization parameters.
    :param render_ax: PyPlot axis on which rendering can take place.
    """
    # Instantiate environment and agents
    env = env_class(**env_kwargs)
    tabular_agent = AUPTabularAgent(env)
    state_Q = (AUPTabularAgent(env, do_state_penalties=True)).penalty_Q
    movies, agents = [], [AUPTabularAgent(env, num_rpenalties=0),  # vanilla
                          AUPAgent(penalty_Q=tabular_agent.penalty_Q),  # full AUP
                          tabular_agent,
                          AUPAgent(penalty_Q=state_Q, baseline='inaction', deviation='decrease'),  # RR
                          AUPAgent(penalty_Q=tabular_agent.penalty_Q, baseline='start'),
                          AUPAgent(penalty_Q=tabular_agent.penalty_Q, baseline='inaction'),
                          AUPAgent(penalty_Q=tabular_agent.penalty_Q, deviation='decrease')
                          ]

    for agent in agents:
        ret, _, perf, frames = run_episode(agent, env, save_frames=True, render_ax=render_ax)
        movies.append((agent.name, frames))
        if hasattr(agent, 'training_performance'):
            print(agent.name, agent.training_performance[0][-1])
        else:
            print(agent.name, perf)

    return movies


games = [#(conveyor.ConveyorBeltEnvironment, {'variant': 'vase'}),
         #(conveyor.ConveyorBeltEnvironment, {'variant': 'sushi'}),
         #(burning.SideEffectsBurningBuildingEnvironment, {'level': 0}),
         #(burning.SideEffectsBurningBuildingEnvironment, {'level': 1}),
         (sokoban.SideEffectsSokobanEnvironment, {'level': 0}),
         (sushi.SideEffectsSushiBotEnvironment, {'level': 0}),
         (vase.SideEffectsVaseEnvironment, {'level': 0}),
         (coffee.SideEffectsCoffeeBotEnvironment, {'level': 0}),
         (survival.SurvivalIncentiveEnvironment, {'level': 0})]

# Plot setup
#plt.switch_backend('TkAgg')
plt.style.use('ggplot')

# Get individual game ablations
for (game, kwargs) in games:
    run_game(game, kwargs)
