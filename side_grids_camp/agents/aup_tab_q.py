from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict, namedtuple
import experiments.environment_helper as environment_helper
import numpy as np


class AUPTabularAgent:  # TODO sokoban, sushi pause, dog, corrigibility
    name = "Tabular AUP"
    epsilon = 0.25  # chance of choosing a random action in training

    def __init__(self, env, N=2, do_state_penalties=False, num_rpenalties=10, discount=.999, num_episodes=600):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param N: How much impact the agent can have.
        :param penalties: Reward functions whose shifts in attainable values will be penalized.
        """
        self.actions = range(env.action_spec().maximum + 1)
        self.discount = discount
        self.num_episodes = num_episodes
        self.N = N

        self.penalties = environment_helper.derive_possible_rewards(env) if do_state_penalties else None
        if num_rpenalties > 0:
            self.penalties = [defaultdict(np.random.random) for _ in range(num_rpenalties)]
            self.penalty_Q = defaultdict(lambda: np.zeros((num_rpenalties, len(self.actions))))
        self.goal_reward = env.GOAL_REWARD

        # Store average stats
        self.training_performance = np.zeros((2, self.num_episodes))
        for penalty_idx in range(len(self.penalties)):
            self.train(env, type=penalty_idx)

        # Train AUP according to the inferred composite reward - (L_1 change in penalty_Q)
        self.train(env, type='AUP')

        env.reset()

    def train(self, env, type='AUP'):
        is_AUP = type == 'AUP'
        num_trials = 1 if is_AUP else 1
        for _ in range(num_trials):
            if is_AUP:
                self.AUP_Q = defaultdict(lambda: np.zeros(len(self.actions)))
            for episode in range(self.num_episodes):
                time_step = env.reset()
                while not time_step.last():
                    last_board = str(time_step.observation['board'])
                    action = self.behavior_action(last_board, type)
                    time_step = env.step(action)
                    self.update_greedy(last_board, action, time_step, is_AUP)

                if is_AUP:
                    ret, _, perf, _ = environment_helper.run_episode(self, env)
                    self.training_performance[0][episode] += ret / num_trials
                    self.training_performance[1][episode] += perf / num_trials

    def act(self, obs):
        return self.AUP_Q[str(obs['board'])].argmax()

    def behavior_action(self, board, type):
        """Returns the e-greedy action for the state board string."""
        if type == 'AUP':
            greedy = self.AUP_Q[board].argmax()
        else:
            greedy = self.penalty_Q[board][type].argmax(axis=0)
        if np.random.random() < self.epsilon or len(self.actions) == 1:
            return greedy
        else:  # choose anything else
            return np.random.choice(self.actions, p=[1.0 / (len(self.actions) - 1) if i != greedy
                                                     else 0 for i in self.actions])

    def get_penalty(self, board, action):
        if len(self.penalties) == 0: return 0
        action_attainable = self.penalty_Q[board][:, action]
        null_attainable = self.penalty_Q[board][:, safety_game.Actions.NOTHING]
        null_sum = sum(abs(null_attainable))

        # Scaled difference between taking action and doing nothing
        return sum(abs(action_attainable - null_attainable)) / (self.N * null_sum) if null_sum \
            else 1.01  # ImpactUnit is 0!

    def update_greedy(self, last_board, action, time_step, train_AUP=False):
        """Perform TD update on observed reward."""
        learning_rate = 1
        new_board = str(time_step.observation['board'])

        def calculate_update(last_board, action, time_step, pen_idx=None):
            """Do the update for the main function (or the penalty function at the given index)."""
            if pen_idx is not None:
                reward = self.penalties[pen_idx][new_board]
                new_Q, old_Q = self.penalty_Q[new_board][pen_idx].max(), \
                               self.penalty_Q[last_board][pen_idx, action]
            else:
                reward = time_step.reward - self.get_penalty(last_board, action)
                new_Q, old_Q = self.AUP_Q[new_board].max(), self.AUP_Q[last_board][action]
            return learning_rate * (reward + self.discount * new_Q - old_Q)

        if not train_AUP:
            # Learn the other reward functions, too
            for pen_idx in range(len(self.penalties)):
                self.penalty_Q[last_board][pen_idx, action] += calculate_update(last_board, action, time_step, pen_idx)
            #self.penalty_Q[last_board][:, action] = np.clip(self.penalty_Q[last_board][:, action], 0, self.goal_reward)
        else:
            self.AUP_Q[last_board][action] += calculate_update(last_board, action, time_step)
