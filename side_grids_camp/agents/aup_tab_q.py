from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict, namedtuple
import numpy as np


class AUPTabularAgent:  # TODO sokoban, sushi pause, dog, corrigibility
    name = "Tabular AUP"
    epsilon = 0.2  # chance of choosing a random action in training

    def __init__(self, env, N=3, penalties=(), discount=.99, num_episodes=700):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param N: How much impact the agent can have.
        :param penalties: Reward functions whose shifts in attainable values will be penalized.
        """
        self.actions = range(env.action_spec().maximum + 1)
        # Store times visited in second spot
        self.Q = defaultdict(lambda: np.zeros((len(self.actions), 2)))
        self.discount = discount
        self.num_episodes = num_episodes
        self.N = N

        self.penalties = penalties  # store the actual penalty functions
        if penalties:
            self.penalty_Q = defaultdict(lambda: np.zeros((len(penalties), len(self.actions))))
        self.goal_reward = env.GOAL_REWARD

        self.train(env)  # let's get to work!

        # Train AUP according to the inferred composite reward - (L_1 change in penalty_Q)
        self.AUP_Q = defaultdict(lambda: np.zeros((len(self.actions), 2)))
        self.train(env, train_AUP=True)

        env.reset()

    def train(self, env, train_AUP=False):
        EpisodeStats = namedtuple("EpisodeStats", ["rewards", "performance"])
        self.training_performance = EpisodeStats(rewards=np.zeros(self.num_episodes),
                                                 performance=np.zeros(self.num_episodes))
        for episode in range(self.num_episodes):
            time_step = env.reset()
            while not time_step.last():
                last_board = str(time_step.observation['board'])
                action = self.behavior_action(last_board, train_AUP)
                time_step = env.step(action)
                self.update_greedy(last_board, action, time_step, train_AUP)

            self.training_performance["rewards"][episode] = env.episode_return
            self.training_performance["performance"][episode] = env._episodic_performances[-1]

    def act(self, obs):
        return self.AUP_Q[str(obs['board'])][:, 0].argmax(axis=0)

    def behavior_action(self, board, train_AUP=False):
        """Returns the e-greedy action for the state board string."""
        greedy = self.AUP_Q[board][:, 0].argmax(axis=0) if train_AUP else self.Q[board][:, 0].argmax(axis=0)
        if np.random.random() < self.epsilon or len(self.actions) == 1:
            return greedy
        else:  # choose anything else
            return np.random.choice(self.actions, p=[1.0 / (len(self.actions) - 1) if i != greedy
                                                     else 0 for i in self.actions])

    def get_penalty(self, board, action):
        action_attainable = self.penalty_Q[board][:, action]
        null_attainable = self.penalty_Q[board][:, safety_game.Actions.NOTHING]
        null_sum = sum(abs(null_attainable))

        # Scaled difference between taking action and doing nothing
        return sum(abs(action_attainable - null_attainable)) / (self.N * null_sum) if null_sum \
            else 1.01  # ImpactUnit is 0!

    def update_greedy(self, last_board, action, time_step, train_AUP=False):
        """Perform TD update on observed reward."""
        learning_rate = 1 #self.update_visited_get_lr(last_board, action, train_AUP)
        new_board = str(time_step.observation['board'])

        def calculate_update(last_board, action, time_step, pen_idx=None, train_AUP=False):
            """Do the update for the main function (or the penalty function at the given index)."""
            if not train_AUP:
                if pen_idx is not None:
                    reward = self.penalties[pen_idx](time_step.observation)
                    new_Q, old_Q = self.penalty_Q[new_board][pen_idx, :].max(), \
                                   self.penalty_Q[last_board][pen_idx, action]
                else:
                    reward = time_step.reward
                    new_Q, old_Q = self.Q[new_board][:, 0].max(), self.Q[last_board][action, 0]
            else:
                reward = time_step.reward - self.get_penalty(last_board, action)
                new_Q, old_Q = self.AUP_Q[new_board][:, 0].max(), self.AUP_Q[last_board][action, 0]
            return learning_rate * (reward + self.discount * new_Q - old_Q)

        if not train_AUP:
            self.Q[last_board][action, 0] += calculate_update(last_board, action, time_step)

            # Learn the other reward functions, too
            for pen_idx in range(len(self.penalties)):
                self.penalty_Q[last_board][pen_idx, action] += calculate_update(last_board, action,
                                                                                   time_step, pen_idx)
            #self.penalty_Q[last_board][:, action] = self.penalty_Q[last_board][:, action, 0],
                                                                # simulate end of level
        else:
            self.AUP_Q[last_board][action, 0] += calculate_update(last_board, action, time_step, train_AUP=train_AUP)

    def update_visited_get_lr(self, last_board, action, train_AUP=False):
        Q = self.Q[last_board] if not train_AUP else self.AUP_Q[last_board]
        Q[action, 1] += 1
        return 1/Q[action, 1]**2

#def train_Q(env, penalties, discount=.99, num_episodes=600)
