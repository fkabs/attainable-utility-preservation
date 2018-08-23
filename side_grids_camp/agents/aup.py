import numpy as np


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'Attainable Utility Preservation'

    def __init__(self, r_A, penalties):
        """

        :param r_A: the agent's original deep Q-function estimator.
        :param penalties: the estimators whose shifts in attainable value will be penalized.
        """
        self.r_A = r_A
        self.penalties = penalties

    def act(self, obs):
        """Act to greedily maximize the penalized reward function."""
        new_q = self.r_A.get_q(obs) - sum([penalty.get_q(obs)
                                            for penalty in self.penalties]) / len(self.penalties)
        return np.argmax(new_q)