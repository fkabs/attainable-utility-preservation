import numpy as np
from ai_safety_gridworlds.environments.shared import safety_game


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'Attainable Utility Preservation'

    def __init__(self, penalties=(), m=9, N=2, impact_pct=1):
        """

        :param penalties: Reward functions whose shifts in attainable values will be penalized.
        :param m: The horizon up to which the agent will calculate attainable utilties after each action.
        :param N: Scale harshness of penalty: 1/N * penalty term.
        :param impact_pct: How much to scale the attainable null penalty by when regularizing (standing in for a_signal).
        """
        self.penalties = penalties
        self.m = m
        self.N = N
        self.impact_unit = impact_pct

        self.attainable = dict()  # memoize the (board, steps_left) reward + penalty values, inclusive of current step
        self.cached_actions = dict()

    def act(self, env, so_far=[]):
        """Get penalties from brute-force search and choose best penalized action.

        :param so_far: The actions up until now; assuming a deterministic environment, this allows cheap restarts.
        """
        penalized_rewards = [self.penalized_reward(env, action, so_far)
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
                r = self.penalized_reward(env, a, so_far)
                if steps_left > 0:
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
        env.reset()
        for action in actions:
            env.step(action)
        return env

    def penalized_reward(self, env, action, so_far=[]):  # TODO heavy penalty for finishing level
        """The penalized reward for taking the given action in the current state. Steps the environment forward.

        :param env: Simulator.
        :param action: The action in question.
        :param so_far: Actions taken up until now.
        """
        time_step = env.step(action)
        reward, scaled_penalty = time_step.reward if time_step.reward else 0, 0
        if self.penalties and action != safety_game.Actions.NOTHING:
            action_pen = self.attainable_penalties(env, self.m, so_far + [action])
            self.restart(env, so_far)
            null_pen = self.attainable_penalties(env, self.m, so_far + [safety_game.Actions.NOTHING])
            null_sum = sum(null_pen)
            self.restart(env, so_far + [action])

            # Scaled difference between taking action and doing nothing
            scaled_penalty = sum(abs(action_pen - null_pen)) / (len(action_pen) * self.N * sum(null_pen)) if null_sum \
                else 0
        return reward - scaled_penalty

    def attainable_penalties(self, env, steps_left, so_far=[]):
        """Returns penalty rewards attainable within steps_left steps.

        :param env: Simulator.
        :param steps_left: Remaining depth.
        :param so_far: Actions taken up until now.
        """
        current_hash = (str(env._last_observations['board']), steps_left)
        if current_hash not in self.attainable:
            pens = np.array([penalty(env._last_observations) for penalty in self.penalties])
            if steps_left == 0 or env._game_over:
                return pens

            # For each penalty function, what's the best we can do from here?
            attainable_penalties = np.full(len(self.penalties), float("-inf"))
            for action in range(env.action_spec().maximum + 1):
                env.step(action)

                # See what reward and penalties we can attain from here
                attainable_penalties = np.maximum(attainable_penalties,
                                                  self.attainable_penalties(env, steps_left - 1, so_far=so_far + [action]))
                self.restart(env, so_far)

            # Make sure attainable penalties aren't double-counting goal attainment
            self.attainable[current_hash] = np.minimum(pens + attainable_penalties,
                                                       np.full(len(pens), (env.GOAL_REWARD + env.MOVEMENT_REWARD)))
        return self.attainable[current_hash]
