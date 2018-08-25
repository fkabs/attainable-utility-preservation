import copy
import numpy as np


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'Attainable Utility Preservation'
    null_action = 4  # TODO make robust

    def __init__(self, penalties, m=7):
        """

        :param penalties: the estimators whose shifts in attainable value will be penalized.
        :param m: the horizon up to which the agent will act after acting.
        """
        self.penalties = penalties
        self.m = m

    def act(self, env):
        """Get penalties from brute-force search and choose best penalized action."""
        return np.argmax(self.penalized_rewards(env))

    def penalized_rewards(self, env):
        rewards, penalty_lsts = np.array([]), []
        for action in range(env.action_spec().maximum + 1):  # TODO use depth first?
            new_env = copy.deepcopy(env)
            time_step = new_env.step(action)
            rewards.append(time_step.reward)
            penalty_lsts.append(np.array(self.attainable_rewards(time_step.get_observation(), new_env, self.m)))

        #
        null = penalty_lsts[self.null_action]
        attainable_differences = np.array([abs(penalty - null) for penalty in penalty_lsts[:-2]]) # our ability to change each utility compared to the null action
        attainable_differences.append(0)
        return rewards - attainable_differences

    def attainable_rewards(self, obs, env, steps_left):
        """Returns best normal and penalty rewards attainable within steps_left steps."""
        pens = [penalty(obs) for penalty in self.penalties]
        if not steps_left:
            return pens

        penalty_lsts = []  # for each action, how well could we accomplish the penalty utility?
        for action in range(env.action_spec().maximum + 1):  # TODO use depth first?
            new_env = copy.deepcopy(env)
            time_step = new_env.step(action)
            penalty_lsts.append(self.attainable_rewards(time_step.get_observation(), new_env, steps_left-1))
        return [pen + max(next_pen) for pen, next_pen in zip(pens, penalty_lsts)]  # care about best from each

    """
    def act(self, obs):
        #Act to greedily maximize the penalized reward function.
        new_q = self.r_A.get_q(obs) - sum([penalty.get_q(obs)
                                            for penalty in self.penalties]) / len(self.penalties)
        return np.argmax(new_q)
    """
