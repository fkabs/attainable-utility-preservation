from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')

import os
import sys
import itertools
from functools import partial
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from environment_helper import *
from ai_safety_gridworlds.environments import *
from ai_safety_gridworlds.environments.shared import safety_game
from agents.model_free_aup import ModelFreeAUPAgent


def make_charts(prefix = '', dir = ''):
    colors = {'box':      [v / 1000. for v in box.GAME_BG_COLOURS[box.BOX_CHR]],
              'dog':      [v / 1000. for v in dog.GAME_BG_COLOURS[dog.DOG_CHR]],
              'survival': [v / 1000. for v in survival.GAME_BG_COLOURS[survival.BUTTON_CHR]],
              'conveyor': [v / 1000. for v in conveyor.GAME_BG_COLOURS[conveyor.OBJECT_CHR]],
              'sushi':    [v / 1000. for v in sushi.GAME_BG_COLOURS[sushi.SUSHI_CHR]]}

    order = ['box', 'dog', 'survival', 'conveyor', 'sushi']
    new_names = ['options', 'damage', 'correction', 'offset', 'interference']
    settings_order = ['discount', 'lambd', 'num_rewards']

    plt.style.use('ggplot')
    fig = plt.figure()
    axs = [fig.add_subplot(3, 1, plot_ind + 1) for plot_ind in range(3)]
    fig.set_size_inches(7, 4, forward=True)
    for plot_ind, keyword in enumerate(settings_order):
        setting = settings[keyword]
        counts = np.load(os.path.join(os.path.dirname(__file__), 'plots', dir, prefix + 'counts-' + keyword + '.npy'),
                         encoding="latin1")[()]

        stride = 3 if keyword == 'num_rewards' else 2
        ax = axs[plot_ind]
        ax.tick_params(axis='x', which='minor', bottom=False)

        ax.set_xlabel(setting['label'])
        if keyword == 'lambd':
            ax.set_ylabel('Trials')
            for key in counts.keys():
                counts[key] = counts[key][::-1]
        x = np.array(range(len(setting['iter'])))

        tick_pos, tick_labels = [], []
        text_ind, text = [], []

        width = .85
        offset = (len(setting['iter']) + 1)

        ordered_counts = [(name, counts[name]) for name in order]
        for x_ind, (game_name, data) in enumerate(ordered_counts):
            tick_pos.extend(list(x + offset * x_ind))
            text_ind.append((len(setting['iter']) -.75) / 2 + offset * x_ind)

            tick_labels.extend([setting['iter_disp'][i] if i % stride == 0 else '' for i in range(len(setting['iter']))])
            if keyword == 'discount':
                text.append(r'$\mathtt{' + new_names[x_ind].capitalize() + '}$')

            for ind, (label, color) in enumerate([("Side effect,\nincomplete", (.3, 0, 0)),
                                                  ("Side effect,\ncomplete", (.65, 0, 0)),
                                                  ("No side effect,\nincomplete", "xkcd:gray"),
                                                  ("No side effect,\ncomplete", (0.0, .624, 0.42))]):
                ax.bar(x + offset * x_ind, data[:, ind], width, label=label, color=color,
                       bottom=np.sum(data[:, :ind], axis=1) if ind > 0 else 0, zorder=3)

        # Wrangle ticks and level labels
        ax.set_xlim([-1, tick_pos[-1] + 1])
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xticks(text_ind, minor=True)
        ax.set_xticklabels(text, minor=True, fontdict={"fontsize": 8})
        for lab in ax.xaxis.get_minorticklabels():
            lab.set_y(1.34)
        ax.tick_params(axis='both', width=.5, labelsize=7)

        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:4][::-1], labels[:4][::-1], fontsize='x-small', loc='upper center', facecolor='white',
               edgecolor='white', ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .97), h_pad=0.15)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', dir, prefix + 'counts.pdf'), bbox_inches='tight')

    # Plot of episodic performance data
    perf = np.load(os.path.join(os.path.dirname(__file__), 'plots', dir, prefix + 'performance' + '.npy'), encoding="latin1")[()]

    eps_fig, eps_ax = plt.subplots()
    eps_fig.set_size_inches(7, 2, forward=True)
    eps_ax.set_xlabel('Episode')
    eps_ax.set_ylabel('Performance')
    eps_ax.set_xlim([-150, 6150])
    eps_ax.set_yticks([-1, 0, 1])

    for ind, name in enumerate(order):
        eps_ax.plot(range(0, len(perf[name][0]) * 10, 10),
                    np.average(perf[name], axis=0), label=r'$\mathtt{' + new_names[ind].capitalize() + '}$',
                    color=colors[name], zorder=3)

    # Mark change in exploration strategy
    eps_ax.axvline(x=4000, color=(.4, .4, .4), zorder=1, linewidth=2, linestyle='--')
    eps_ax.legend(loc='upper center', facecolor='white', edgecolor='white', ncol=len(order),
                  bbox_to_anchor=(0.5, 1.2))

    eps_fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', dir, prefix + 'episodes.pdf'), bbox_inches='tight')
    
    # plt.show()


def gen_exps(settings, games):
    experiments = dict()
    
    for (keyword, setting) in settings.items():
        permuations = list(itertools.product(
            list(games.values()),
            setting['iter']
        ))
        
        experiments.update({keyword: permuations})
    
    return experiments


def run_exp(keyword, vaup, exp):
    game, game_kwargs = exp[0]
    iter = exp[1]
    res = {game.name : {}}
    
    print(keyword + ' --> ' + game.name + ': iter ' + str(iter))
    
    env = game(**game_kwargs)
    model_free = ModelFreeAUPAgent(env, trials = 50, vaup = vaup, **{keyword: iter})
    
    # res[game.name].update({'perf' : {list(settings[keyword]['iter']).index(iter) : model_free.performance}})
    if keyword == 'lambd' and iter == ModelFreeAUPAgent.default['lambd']:
        res[game.name].update({'perf' : model_free.performance})
    res[game.name].update({'counts' : {list(settings[keyword]['iter']).index(iter) : model_free.counts[:]}})
    
    return res


if __name__ == '__main__':
    # set number of usable CPU cores
    NUM_CORES = mp.cpu_count()
    
    # settings to test for
    settings = {
        'discount': {
            'label': r'$\gamma$',
            'iter': [1 - 2 ** (-n) for n in range(3, 11)],
            'iter_disp': ['{0:0.3f}'.format(1 - 2 ** (-n)).lstrip("0") for n in range(3, 11)] 
        },
        'lambd': {
            'label': r'$\lambda$',
            'iter': 1/np.arange(.001, 3.001, .3),
            'iter_disp': ['{0:0.1f}'.format(round(l, 1)).lstrip("0") for l in 1/np.arange(.001, 3.001, .3)][::-1] 
        },
        'num_rewards': {
            'label': r'$|\mathcal{R}|$',
            'iter': range(0, 50, 5),
            'iter_disp': range(0, 50, 5) 
        }
    }

    # games/environments to test
    games = {
        'box': (box.BoxEnvironment, {'level': 0}),
        'dog': (dog.DogEnvironment, {'level': 0}),
        'survival': (survival.SurvivalEnvironment, {'level': 0}),
        'conveyor': (conveyor.ConveyorEnvironment, {'variant': 'vase'}),
        'sushi': (sushi.SushiEnvironment, {'level': 0})
    }
    
    # set aup variants to test
    vaups = [None, 'zero', 'avg', 'avg-oth', 'adv', 'rand']
    
    for action_driven in [False, True]:
        # no no-op action for vaup variants
        if action_driven:
            vaups = vaups[1:]
            safety_game.AGENT_LAST_ACTION = 3
        
        # run experiments
        for vaup in vaups:
            prefix = 'aup_' if vaup is None else vaup + '_'
            dir = 'actd' if action_driven else 'noop'
            
            for (keyword, exp) in gen_exps(settings, games).items():        
                # dicts to store agents results, counts and performances
                res, counts, perf = dict(), dict(), dict()

                # reset counts
                for (game, _) in games.values():
                    counts[game.name] = np.zeros((len(settings[keyword]['iter']), 4))
                    
                # distribute experiment-permutations on all cores
                func = partial(run_exp, keyword, vaup)
                pool = mp.Pool(NUM_CORES-1)
                results = pool.map(func, exp)
                
                # set counts and perf based on results
                for res in results:
                    game = list(res.keys())[0]
                    idx = list(res[game]['counts'].keys())[0]
                    counts[game][idx, :] = res[game]['counts'][idx]
                    
                    if keyword == 'lambd' and 'perf' in list(res[game].keys()):
                        perf[game] = res[game]['perf']
                    
                # save results to disk
                if keyword == 'lambd':
                    np.save(os.path.join(os.path.dirname(__file__), 'plots', dir, prefix + 'performance'), perf)
                
                np.save(os.path.join(os.path.dirname(__file__), 'plots', dir, prefix + 'counts-' + keyword), counts)

            make_charts(prefix, dir)
