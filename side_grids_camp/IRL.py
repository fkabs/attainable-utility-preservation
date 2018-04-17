"""
Thanks to:
@misc{alger16,
  author       = {Matthew Alger},
  title        = {Inverse Reinforcement Learning},
  year         = 2016,
  doi          = {10.5281/zenodo.555999},
  url          = {https://doi.org/10.5281/zenodo.555999}
}

and Dima Krasheninnikov

"""
import numpy as np

"""
# trajectories =  array of demonstration trajectories, each trajectory is an array of state action pairs


# feature_matrix = array of feature vectors, each feature vector is associated with the state at that index


# transition_probabilities = probability of moving from one state to another given action, 1 for legal moves, 0 for illegal moves

# feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?

"""


N_ACTIONS = 4
N_STATES = # TODO
LEARNING_RATE = 0.01
EPOCHS = 200
DISCOUNT = 0.01

irl = maxEntIRL(feature_matrix, trans_probs, trajectories)


# TODO methods to map from state to state index and back
# How are states represented here - location of agent and box? Hard to apply in complex envs. Feature array? Greyscale images?

def stateToInt(state):
    # TODO
    # Return int


def getExampleTrajectories():

    trajectories = []
    # TODO
    # (each state in traj is state action pair (state, action)
    # Could generate (take num, len, policy)or do by hand here
    return trajectories



def getFeatureMatrix(states, feature_vectors):

    feature_matrix = []
    # TODO
    # Each state has associated feature vector
    # for n in range (number of possible states)
        # append feature vector
    return feature_matrix


def getTransitionProbabilities(gridworld_environment):

    trans_probs = []
    # for each state
        # for each possible action
            # get state

    return trans_probs


def getFeatureExpectations(feature_matrix, trajectories):

    # feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations


def getSVF(trajectories):
    # State visitiation frequencies

    svf = np.zeros(n_states)

        for trajectory in trajectories:
            for state, _, _ in trajectory:
                svf[state] += 1

        svf /= trajectories.shape[0]

    return svf


def getExpectedSVF(rewards, transition_probability, trajectories):
    # state visitation frequency
    # policy = findPolicy(transition_probability, rewards)

    #

    return expected_svf



def findPolicy(transition_probabilities, rewards, discount_factor):
    """Computes the optimal policy for a given transition probability and reward
    specification by first computing the value function and then taking greedy
    actions based on this.

    Reference code:
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/value_iteration.py
    """
    conv_threshold = 1e-4
    # pull this out to irl method?
    v = getOptimalValueFunction(transition_probabilities,
                                reward,
                                discount_factor,
                                conv_threshold)

    n_states, n_actions, _ = transition_probabilities.shape
    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (rewards[k] + discount_factor * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy


def getOptimalValueFunction(transition_probabilities, reward, discount_factor,
                            conv_threshold):
    """Iterates over states s performing policy evaluation with the standard
    Bellman backup equation for current policy \pi:

    V(s) <- \sum_{a} \pi(a|s) * \sum_{s', r} p(s', r|s, a)[r + \gamma V(s')]

    where a is action, s is current state, s' is the next state, p is the
    transition measure and \gamma is the discount factor (see Sutton & Barto v2,
    page 75).

    Currently this code assumes deterministic transitions and a greedy policy,
    this could be relaxed by implementing other policy choices.

    Args:
        transition_probabilities: (n_states, n_actions, n_states) array
        reward: (n_states) array containing rewards for each state
        discount_factor: float in [0,1]
        conv_threshold: float setting convergence threshold

    Returns a vector of values of length n_states. The following code was used
    as a reference:
    https://github.com/krasheninnikov/max-causal-ent-irl/blob/master/value_iter_and_policy.py
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
    """

    n_states, n_actions, _ = transition_probabilities.shape
    val_func = np.copy(rewards) # initialise value at rewards

    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)

        Q = reward + gamma * np.dot(transition_probabilities, V_prev)
        V = np.amax(Q, axis=1)

        diff = np.amax(abs(V_prev - V))

    return V

def maxEntIRL(feature_matrix, trans_probs, trajectories):

    # TODO

    # Initialise weights = numpy.random.uniform(size=(num_of_states,))

    # Get feature matrix using all states and feature vectors

    # Get expert demo trajectories

    # Get feature expectations


    # for i in range EPOCHS
        # rewards = feature_matrix.dot(weights)

        # get expected svfs (state visitation frequencies)

        # rewards += learning_rate * (feature_expectations - (feature_matrix * expectedsvf))

    # return feature_matrix * rewards





    return rewards
