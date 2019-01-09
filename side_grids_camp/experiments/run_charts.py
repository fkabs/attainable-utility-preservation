from __future__ import print_function
from ai_safety_gridworlds.environments import side_effects_burning_building as burning, side_effects_sokoban as sokoban, \
    side_effects_sushi_bot as sushi, side_effects_vase as vase, survival_incentive as survival, \
    side_effects_conveyor_belt as conveyor, side_effects_coffee_bot as coffee
from agents.aup_tab_q import AUPTabularAgent
from environment_helper import *
import  os
import matplotlib.pyplot as plt

# Plot setup
plt.style.use('ggplot')

games = [#(sokoban.SideEffectsSokobanEnvironment, {'level': 0}),
         (sushi.SideEffectsSushiBotEnvironment, {'level': 0}),
         #(coffee.SideEffectsCoffeeBotEnvironment, {'level': 0}),
         #(survival.SurvivalIncentiveEnvironment, {'level': 0})
        ]

r_discount, r_budget, r_random = False, True, False
# Discount
if r_discount:
    print('discount')
    discounts = [1 - 2**(-n) for n in range(2, 10)]
    fig, ax = plt.subplots()
    ax.set_ylim([-1, 1])

    ax.set_xlabel('Discount')

    for (game, kwargs) in games:
        stats = []
        for discount in discounts:
            env = game(**kwargs)
            tabular_agent = AUPTabularAgent(env, discount=discount)
            stats.append(np.average(tabular_agent.performance[:, -1]))
            print(discount, stats[-1])
        print(game.name, stats)
        ax.plot(discounts, stats, label=game.name, marker='^')

    ax.set_xscale('logit')
    ax.spines['bottom']._adjust_location()
    ax.set_xticklabels(discounts)
    ax.set_xticklabels([], minor=True)
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'discount.pdf'), bbox_inches='tight')

# N
if r_budget:
    print('budget')
    budgets = np.arange(0, 300, 30)
    fig, ax = plt.subplots()
    ax.set_ylim([-1, 1])
    ax.set_xlabel('N')

    x = range(0, AUPTabularAgent.default['episodes'], 10)
    eps_fig, eps_ax = plt.subplots()
    eps_ax.set_xlabel('Episode')

    eps_ax.set_ylim([-2, 1])

    for (game, kwargs) in games:
        stats = []
        for budget in [90]: #budgets:

            env = game(**kwargs)
            tabular_agent = AUPTabularAgent(env, N=budget)  # TODO rerun N=0
            _, _, perf, _ = run_episode(AUPAgent(penalty_Q=tabular_agent.penalty_Q), env)
            #if budget == AUPTabularAgent.default['N']:
            eps_ax.plot(x, np.average(tabular_agent.performance, axis=0), label=game.name + str(budget))
            stats.append(np.average(tabular_agent.performance[:, -1]))
            print(budget, stats[-1], perf)
        print(game.name, stats)
        #ax.plot(budgets, stats, label=game.name, marker='^')

    #fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'N.pdf'), bbox_inches='tight')

    eps_ax.legend(loc=4)
    eps_ax.axvline(x=AUPTabularAgent.default['episodes'] * .4, color='gray')
    #eps_fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'episodes.pdf'), bbox_inches='tight')
    plt.show()

if r_random: # rand pen
    print('random')
    nums = range(0, 30, 3)
    fig, ax = plt.subplots()

    ax.set_ylim([-1, 1])
    ax.set_xlabel('Number of Random Reward Functions')

    for (game, kwargs) in games:
        stats = []
        for num in nums:
            env = game(**kwargs)
            tabular_agent = AUPTabularAgent(env, num_rpenalties=num)
            stats.append(np.average(tabular_agent.performance[:, -1]))
            print(num, stats[-1])
        print(game.name, stats)
        ax.plot(nums, stats, label=game.name, marker='^')
    ax.legend(loc=4)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'num_rewards.pdf'), bbox_inches='tight')
    plt.show()