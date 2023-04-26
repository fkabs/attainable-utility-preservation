from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict
import experiments.environment_helper as environment_helper
import numpy as np
import random
import os
from functools import partial


class ModelFreeAUPAgent:
    name = "Model-free AUP"
    pen_epsilon, AUP_epsilon = .2, .9  # chance of choosing greedy action in training
    default = {'lambd': 1./1.501, 'discount': .996, 'rpenalties': 30, 'episodes': 6000}

    def __init__(self, env, lambd=default['lambd'], state_attainable=False, num_rewards=default['rpenalties'],
                 discount=default['discount'], episodes=default['episodes'], trials=1, use_scale=False, vaup=None):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param lambd: Impact tuning parameter.
        :param state_attainable: True - generate state indicator rewards; false - random rewards.
        :param num_rewards: Size of the attainable set, |\mathcal{R}|.
        :param discount: Discount factor
        :param episodes: Number of episodes
        :param trials: Number of trials
        :param vaup: Use variational AUP (penalize using different strategies)
        """
        
        self.actions = range(env.action_spec().maximum + 1)
        self.probs = [[1.0 / (len(self.actions) - 1) if i != k else 0 for i in self.actions] for k in self.actions]
        self.discount = discount
        self.episodes = episodes
        self.trials = trials
        self.lambd = lambd
        self.state_attainable = state_attainable
        self.use_scale = use_scale
        self.vaup = vaup
        
        if self.vaup != None:
            self.name = 'Baseline (' + self.vaup + ')'

        if state_attainable:
            self.name = 'Relative reachability'
            self.attainable_set = environment_helper.derive_possible_rewards(env)
        else:
            self.attainable_set = [defaultdict(np.random.random) for _ in range(num_rewards)]

        if len(self.attainable_set) == 0:
            self.name = 'Q-learning'  # no penalty applied!
        
        log_type = 'ablation' if trials == 1 else 'counts'
        env_type = 'actd' if len(self.actions) < 5 else 'noop'
        sub_dir = log_type + '/' + env_type
        self.log_dir = '/mnt/rl-reward/fk/attainable-utility-preservation/experiments/q_logs/' + sub_dir

        self.train(env)

    def train(self, env):
        self.performance = np.zeros((self.trials, self.episodes / 10))

        # 0: high-impact, incomplete; 1: high-impact, complete; 2: low-impact, incomplete; 3: low-impact, complete
        self.counts = np.zeros(4)
        
        # dict for storing Q-values
        log = {}

        for trial in range(self.trials):
            self.attainable_Q = defaultdict(partial(np.zeros, (len(self.attainable_set), len(self.actions))))
            # self.attainable_Q = defaultdict(lambda: np.zeros((len(self.attainable_set), len(self.actions))))
            self.AUP_Q = defaultdict(partial(np.zeros, len(self.actions)))
            # self.AUP_Q = defaultdict(lambda: np.zeros(len(self.actions)))
            if not self.state_attainable:
                self.attainable_set = [defaultdict(np.random.random) for _ in range(len(self.attainable_set))]
            self.epsilon = self.pen_epsilon

            for episode in range(self.episodes):
                if episode > 2.0 / 3 * self.episodes:  # begin greedy exploration
                    self.epsilon = self.AUP_epsilon
                time_step = env.reset()
                while not time_step.last():
                    last_board = str(time_step.observation['board'])
                    action = self.behavior_action(last_board)
                    time_step = env.step(action)
                    self.update_greedy(last_board, action, time_step)
                if episode % 10 == 0:
                    _, actions, self.performance[trial][episode / 10], _ = environment_helper.run_episode(self, env)
            self.counts[int(self.performance[trial, -1]) + 2] += 1  # -2 goes to idx 0
            
            log.update({trial: {'aup_q' : self.AUP_Q, 'attainable_q' : self.attainable_Q}})

        env.reset()
        
        if not os.path.exists(self.log_dir + '/' + env.name):
            os.makedirs(self.log_dir + '/' + env.name)
        
        if self.vaup != None:
            np.save(self.log_dir + '/' + env.name + '/' + self.name, log)
    
    def act(self, obs):
        return self.AUP_Q[str(obs['board'])].argmax()

    def behavior_action(self, board):
        """Returns the e-greedy action for the state board string."""
        greedy = self.AUP_Q[board].argmax()
        if np.random.random() < self.epsilon or len(self.actions) == 1:
            return greedy
        else:  # choose anything else
            return np.random.choice(self.actions, p=self.probs[greedy])

    def get_penalty(self, board, action):
        if len(self.attainable_set) == 0: return 0
        
        action_attainable = self.attainable_Q[board][:, action]
        
        if self.vaup == 'adv':
            pi_attainable = np.zeros((len(self.attainable_set), len(self.actions)), dtype = float)
            pi_attainable[:] = self.epsilon / len(self.actions)
            ri_idx = np.arange(len(self.attainable_set))
            a_idx = np.argmax(self.attainable_Q[board], axis = 1)
            pi_attainable[ri_idx, a_idx] = 1 - self.epsilon + self.epsilon / len(self.actions)
            null_attainable = np.sum(self.attainable_Q[board][:] * pi_attainable, axis = 1)
            diff = abs(action_attainable - null_attainable)
        elif self.vaup == 'avg':
            null_attainable = np.mean(self.attainable_Q[board][:], axis = 1)
            diff = abs(action_attainable) - abs(null_attainable)
        elif self.vaup == 'avg-oth':
            null_attainable = np.mean(self.attainable_Q[board][:, tuple(filter(lambda a: a != action, self.actions))], axis = 1)
            diff = abs(action_attainable) - abs(null_attainable)
        elif self.vaup == 'rand':
            ri_idx = np.arange(len(self.attainable_set))
            a_idx = np.random.choice([a for a in self.actions if a != action], len(self.attainable_set))
            null_attainable = self.attainable_Q[board][ri_idx, a_idx]
            diff = abs(action_attainable - null_attainable)
        elif self.vaup == 'zero':
            null_attainable = np.zeros(len(action_attainable))
            diff = abs(action_attainable - null_attainable)
        else:
            null_attainable = self.attainable_Q[board][:, safety_game.Actions.NOTHING]
            diff = abs(action_attainable - null_attainable)

        # Scaling number or vector (per-AU)
        if self.use_scale:
            scale = sum(abs(null_attainable))
            if scale == 0:
                scale = 1
            penalty = sum(diff / scale)
        else:
            scale = np.copy(null_attainable)
            scale[scale == 0] = 1  # avoid division by zero
            penalty = np.average(np.divide(diff, scale))
        
        # Scaled difference between taking action and baseline
        return self.lambd * penalty  # ImpactUnit is 0!

    def update_greedy(self, last_board, action, time_step):
        """Perform TD update on observed reward."""
        learning_rate = 1
        new_board = str(time_step.observation['board'])

        def calculate_update(attainable_idx=None):
            """Do the update for the main function (or the attainable function at the given index)."""
            if attainable_idx is not None:
                reward = self.attainable_set[attainable_idx](new_board) if self.state_attainable \
                    else self.attainable_set[attainable_idx][new_board]
                new_Q, old_Q = self.attainable_Q[new_board][attainable_idx].max(), \
                                self.attainable_Q[last_board][attainable_idx, action]
            else:
                reward = time_step.reward - self.get_penalty(last_board, action)
                new_Q, old_Q = self.AUP_Q[new_board].max(), self.AUP_Q[last_board][action]
            return learning_rate * (reward + self.discount * new_Q - old_Q)

        # Learn the attainable reward functions
        for attainable_idx in range(len(self.attainable_set)):
            self.attainable_Q[last_board][attainable_idx, action] += calculate_update(attainable_idx)
        
        if self.state_attainable:
            self.attainable_Q[last_board][:, action] = np.clip(self.attainable_Q[last_board][:, action], 0, 1)
        
        self.AUP_Q[last_board][action] += calculate_update()
