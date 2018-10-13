import numpy as np
import os
import pickle
from ai_safety_gridworlds.environments.shared import safety_game


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """

    def __init__(self, penalty_Q, N=2, save_dir=None):
        """

        :param penalties: Reward functions whose shifts in attainable values will be penalized.
        :param N: Scale harshness of penalty: 1/N * penalty term.
        :param save_dir: The directory from which the memoized data are loaded. Change when reparametrizing.
        """
        self.penalty_Q = penalty_Q
        self.N = N
        self.name = "AUP"

        try:
            self.dir = os.path.join(save_dir, self.name)
            with open(os.path.join(self.dir, "cached.pkl"), 'rb') as c:
                self.cached_actions = pickle.load(c)
        except:
            self.cached_actions = dict()

    def act(self, env, so_far=[]):
        """Get penalties from brute-force search and choose best penalized action.

        :param so_far: The actions up until now; assuming a deterministic environment, this allows cheap restarts.
        """
        penalized_rewards = [self.penalized_reward(env, action, so_far)[0]
                             for action in range(env.action_spec().maximum + 1)]
        return np.argmax(penalized_rewards)

    def get_actions(self, env, steps_left, so_far=[]):
        """Figure out the n-step optimal plan, returning it and its return.

        :param env: Simulator.
        :param steps_left: >= 1; how many steps to plan over.
        :param so_far: actions taken up until now (used for restart).
        """
        current_hash = (str(env.last_observations['board']), steps_left)
        if current_hash not in self.cached_actions:
            best_actions, best_ret = [], float('-inf')
            for a in range(env.action_spec().maximum + 1):
                r, done = self.penalized_reward(env, a, steps_left, so_far)
                if steps_left > 0 and not done:
                    actions, ret = self.get_actions(env, steps_left-1, so_far + [a])
                else:
                    actions, ret = [], 0
                if r + ret > best_ret:
                    best_actions, best_ret = [a] + actions, r + ret
                self.restart(env, so_far)

            self.cached_actions[current_hash] = best_actions, best_ret
        return self.cached_actions[current_hash]

    @staticmethod
    def restart(env, actions):
        """Reset the environment and return the result of executing the action sequence."""
        time_step = env.reset()
        for action in actions:
            if time_step.last(): break
            time_step = env.step(action)

    def penalized_reward(self, env, action, steps_left, so_far=[]):
        """The penalized reward for taking the given action in the current state. Steps the environment forward.

        :param env: Simulator.
        :param action: The action in question.
        :param steps_left: How many steps are left in the plan.
        :param so_far: Actions taken up until now.
        :returns penalized_reward:
        :returns is_last: Whether the episode is terminated.
        """
        time_step = env.step(action)
        reward, scaled_penalty = time_step.reward if time_step.reward else 0, 0
        for _ in range(steps_left-1):
            if time_step.last(): break
            time_step = env.step(safety_game.Actions.NOTHING)
        if self.penalty_Q and action != safety_game.Actions.NOTHING:
            action_plan, inaction_plan = so_far + [action] + [safety_game.Actions.NOTHING] * (steps_left - 1), \
                                         so_far + [safety_game.Actions.NOTHING] * steps_left
            self.restart(env, action_plan)
            action_attainable = self.penalty_Q[str(env._last_observations['board'])]
            self.restart(env, inaction_plan)
            null_attainable = self.penalty_Q[str(env._last_observations['board'])]
            null_sum = sum(abs(null_attainable))
            self.restart(env, so_far + [action])

            # Scaled difference between taking action and doing nothing
            scaled_penalty = sum(abs(action_attainable - null_attainable)) / (self.N * null_sum) if null_sum \
                else 1.01  # ImpactUnit is 0!
        return reward - scaled_penalty, time_step.last()
