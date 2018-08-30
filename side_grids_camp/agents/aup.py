import numpy as np
from ai_safety_gridworlds.environments.shared import safety_game


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'Attainable Utility Preservation'

    def __init__(self, penalties=(), m=8, N=2, impact_unit=1):
        """

        :param penalties: reward functions whose shifts in attainable values will be penalized.
        :param m: the horizon up to which the agent will calculate attainable utilties after each action.
        :param N: up to this many of ImpactUnit is allowed.
        """
        self.penalties = penalties
        self.m = m
        self.N = N
        self.impact_unit = impact_unit

        self.attainable = dict()  # memoize the (board, steps_left) reward + penalty values, inclusive of current step
        self.cached_actions = dict()

    def act(self, env, actions=[]):
        """Get penalties from brute-force search and choose best penalized action.

        :param actions: the actions up until now (assuming a deterministic environment, this allows cheap restarts)
        """
        penalized_rewards = self.penalized_rewards(env, actions)
        return np.argmax(penalized_rewards)

    def get_actions(self, env, steps_left, so_far=[]):
        """Figure out the n-step optimal plan.

        :param env: Simulator.
        :param steps_left: >= 1; how many steps to plan over.
        :param so_far: actions taken up until now (used for restart).
        """
        if env._game_over:
            return [], 0

        current_hash = (str(env.last_observations['board']), steps_left)
        if current_hash not in self.cached_actions:
            pen_rewards = self.penalized_rewards(env, so_far)
            if steps_left == 1:
                return [np.argmax(pen_rewards)], max(pen_rewards)

            # If not last action, figure out best we can do for penalized rewards
            best_actions, best_v = [], float('-inf')
            for a in range(env.action_spec().maximum + 1):
                env.step(a)

                actions, v = self.get_actions(env, steps_left-1, so_far + [a])
                if pen_rewards[a] + v > best_v:
                    best_actions, best_v = [a] + actions, pen_rewards[a] + v
                self.restart(env, so_far)

            self.cached_actions[current_hash] = best_actions, best_v
        return self.cached_actions[current_hash]   # TODO figure out why won't go around block?
    # condition: len(so_far) == 2 and so_far[0] == 2 and so_far[1] == 1

    @staticmethod
    def restart(env, actions):
        """Reset the environment and return the result of executing the action sequence."""
        env.reset()
        for action in actions:
            env.step(action)
        return env

    def penalized_rewards(self, env, so_far=[]):
        """The penalized rewards (according to attainable penalty terms) after each action in the current state.

        :param env: Simulator.
        :param so_far: Actions taken up until now.
        """
        rewards, penalties = np.zeros(env.action_spec().maximum + 1), \
                             np.zeros((env.action_spec().maximum + 1, len(self.penalties)))

        for action in range(env.action_spec().maximum + 1):  # non-null actions
            env.step(action)
            rewards[action] = env._last_reward
            # Attainable penalties within m steps after acting
            penalties[action][:] = self.attainable_penalties(env, self.m, so_far=so_far + [action])

        # Difference of attainable rewards between taking action and doing nothing
        action_differences = np.array([abs(penalty - penalties[safety_game.Actions.NOTHING])
                                       for penalty in penalties])
        weighted_penalties = np.array([sum(diffs) / len(diffs)
                                       for diffs in action_differences]) / (self.N * self.impact_unit) if self.penalties \
            else np.zeros(len(rewards))
        return rewards - weighted_penalties

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
