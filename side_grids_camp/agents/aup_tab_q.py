from collections import defaultdict
import numpy as np


class AUPTabularAgent:
    name = "Tabular AUP"
    discount = 1  # how much it cares about future rewards
    epsilon = 0.2  # chance of choosing a random action in training
    num_episodes = 300

    def __init__(self, env, penalties=()):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy."""
        self.actions = range(env.action_spec().maximum + 1)
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))

        self.train(env)  # let's get to work!
        env.reset()

    def train(self, env):
        for episode in range(self.num_episodes):  # run the appropriate number of episodes
            time_step = env.reset()
            while not time_step.last():
                last_board = str(time_step.observation['board'])
                action = self.behavior_action(last_board)
                time_step = env.step(action)
                self.update_greedy(last_board, action, time_step)

    def greedy_a(self, board):
        """Get the greedy action for the board string."""
        return self.Q[board].argmax(axis=0)

    def act(self, obs):
        return self.greedy_a(str(obs['board']))

    def behavior_action(self, board):
        """Returns the e-greedy action for the state board string."""
        greedy = self.greedy_a(board)
        if np.random.random() < self.epsilon or len(self.actions) == 1:
            return greedy
        else:  # choose anything else
            return np.random.choice(self.actions, p=[1.0 / (len(self.actions) - 1) if i != greedy
                                                     else 0 for i in self.actions])

    def update_greedy(self, last_board, action, time_step):
        """Perform TD update on observed reward."""
        learning_rate = 1
        new_board = str(time_step.observation['board'])
        self.Q[last_board][action] += learning_rate * (time_step.reward + self.discount * self.Q[new_board].max()
                                                      - self.Q[last_board][action])
