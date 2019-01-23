from __future__ import print_function
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_sushi_bot as sushi, side_effects_vase as vase, survival_incentive as survival, \
    side_effects_conveyor_belt as conveyor, side_effects_coffee_bot as coffee
from agents.aup_tab_q import AUPTabularAgent
from environment_helper import *
import  os
import matplotlib.pyplot as plt
from multiprocessing import Pool

settings = [{'label': 'Discount', 'iter': [1 - 2**(-n) for n in range(3, 11)], 'keyword': 'discount'},
            {'label': 'N', 'iter': np.arange(0, 300, 30), 'keyword': 'N'},
            {'label': 'Number of Random Reward Functions', 'iter': range(0, 50, 5), 'keyword': 'num_rewards'}]

games = [#(sokoban.SideEffectsSokobanEnvironment, {'level': 0}),
         #(sushi.SideEffectsSushiBotEnvironment, {'level': 0}),
         (conveyor.ConveyorBeltEnvironment, {'variant': 'vase'}),
         #(survival.SurvivalIncentiveEnvironment, {'level': 0})
        ]


def run_exp(ind):
    setting = settings[ind]
    print(setting['label'])

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set_ylim([-2, 1])
    ax.set_xlabel(setting['label'])
    ax.legend(loc=4)
    if setting['label'] == 'N':
        eps_fig, eps_ax = plt.subplots()
        eps_ax.set_xlabel('Episode')
        eps_ax.set_ylim([-2, 1])

    counts = dict()
    for (game, kwargs) in games:
        counts[game.name] = np.zeros((len(setting['iter']), 4))
        for (idx, item) in enumerate(setting['iter']):
            env = game(**kwargs)
            tabular_agent = AUPTabularAgent(env, trials=2, **{setting['keyword']: item})
            if setting['label'] == 'N' and item == AUPTabularAgent.default['N']:
                np.save(os.path.join(os.path.dirname(__file__), 'plots', 'performance3-' + game.name),
                        tabular_agent.performance)
                eps_ax.plot(range(0, AUPTabularAgent.default['episodes'], 10),
                            np.average(tabular_agent.performance, axis=0), label=game.name.capitalize())
            counts[game.name][idx, :] = tabular_agent.counts[:]
            print(setting['keyword'], item, tabular_agent.counts)
        print(game.name.capitalize())
        #ax.plot(setting['iter'], stats, label=game.name.capitalize(), marker='^')
    np.save(os.path.join(os.path.dirname(__file__), 'plots', 'counts3-' + setting['keyword']), counts)

    if setting['label'] == 'N':
        eps_ax.legend(loc=4)
        eps_ax.axvline(x=AUPTabularAgent.default['episodes'] * 2/3, color='gray')
        eps_fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'episodes3.pdf'), bbox_inches='tight')
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', setting['keyword'] + '3.pdf'),
                bbox_inches='tight')


if __name__ == '__main__':
    run_exp(1)
    #p = Pool(3)
    #p.map(run_exp, range(len(settings)))
