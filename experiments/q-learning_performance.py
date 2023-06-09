from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')

import os
import warnings
import datetime
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment_helper import *
from ai_safety_gridworlds.environments import *
from agents.model_free_aup import ModelFreeAUPAgent


def make_charts():
    plt.style.use('ggplot')
    
    env_names = ['box', 'dog', 'survival', 'conveyor', 'sushi']
    env_new_names = ['options', 'damage', 'correction', 'offset', 'interference']
    
    agents = {}
    
    for env_variant in os.listdir(os.path.join(os.path.dirname(__file__), 'plots')):
        agents[env_variant] = {}
        for agent in os.listdir(os.path.join(os.path.dirname(__file__), 'plots', env_variant)):
            if agent.endswith('.npy'):
                agents[env_variant][agent.split('_')[0]] = np.load(
                    file = os.path.join(os.path.dirname(__file__), 'plots', env_variant, agent),
                    encoding = 'latin1',
                    allow_pickle = True
                )[()]
    
    for env_variant in list(agents.keys()):
        q_learning = agents[env_variant]['q-learning']
        
        for agent, perf in agents[env_variant].items():
            if agent == 'q-learning':
                continue
            
            for ind, name in enumerate(env_names):
                eps_fig, eps_ax = plt.subplots()
                eps_fig.set_size_inches(7, 2, forward = True)
                eps_ax.set_xlabel('Episode')
                eps_ax.set_ylabel('Performance')
                eps_ax.set_xlim([-150, 6150])
                eps_ax.set_yticks([-1, 0, 1])
                
                # q-learning
                eps_ax.plot(
                    range(0, len(q_learning[name][0]) * 10, 10),
                    np.average(q_learning[name], axis = 0),
                    label = 'Q-learning',
                    zorder = 3
                )
                
                # agent
                eps_ax.plot(
                    range(0, len(perf[name][0]) * 10, 10),
                    np.average(perf[name], axis = 0),
                    label = f'Baseline ({agent})',
                    zorder = 3
                )
                
                # show plots y-axis between -1 and 1
                eps_ax.set_ylim([-2, 1.1])
            
                eps_ax.axvline(x = 4000, color = (.4, .4, .4), zorder = 1, linewidth = 2, linestyle = '--')
                eps_ax.legend(loc = 'upper center', facecolor = 'white', edgecolor = 'white', ncol = len(env_names), bbox_to_anchor = (0.5, 1.2))
            
                save_path = os.path.join(
                    os.path.dirname(__file__),
                    'plots',
                    env_variant, env_new_names[ind] + '_' + agent + '_vs_q-learning.pdf'
                )
                eps_fig.savefig(save_path, bbox_inches = 'tight')
                plt.close(eps_fig)


def run_game(env_variant, game):
    env_class, env_kwargs = game
    agent = ModelFreeAUPAgent(env_class(**env_kwargs), num_rewards = 0, trials = 50)  # vanilla (standard q-learner)
    
    return agent.performance


if __name__ == '__main__':
    # set number of usable CPU cores
    NUM_CORES = mp.cpu_count()

    # Plot setup
    plt.style.use('ggplot')
    
    # parameter for action-driven environments
    env_variants = ['noop', 'actd']
    
    for env_variant in env_variants:
        # no no-op action for vaup variants
        if env_variant == 'actd':
            safety_game.AGENT_LAST_ACTION = 3
        else:
            safety_game.AGENT_LAST_ACTION = 4
        
        games = [
            (box.BoxEnvironment, {'level': 0}),
            (dog.DogEnvironment, {'level': 0}),
            (survival.SurvivalEnvironment, {'level': 0}),
            (conveyor.ConveyorEnvironment, {'variant': 'vase'}),
            (sushi.SushiEnvironment, {'level': 0})
            # (conveyor.ConveyorEnvironment, {'variant': 'sushi'}),
            # (vase.VaseEnvironment, {'level': 0}),
            # (burning.BurningEnvironment, {'level': 0}),
            # (burning.BurningEnvironment, {'level': 1})
        ]
        
        print('-'*32)
        print(env_variant.upper())
        print('-'*32)
        print('\n')
        
        res = dict()
        for game in games:
            perf = run_game(env_variant, game)
            res.update({game[0].name : perf})
        
        np.save(os.path.join(os.path.dirname(__file__), 'plots', env_variant, 'q-learning_performance'), res)
    
    make_charts()
