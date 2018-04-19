"""
Major thanks to:

@misc{alger16,
  author       = {Matthew Alger},
  title        = {Inverse Reinforcement Learning},
  year         = 2016,
  doi          = {10.5281/zenodo.555999},
  url          = {https://doi.org/10.5281/zenodo.555999}
}

and Dima Krasheninnikov.

"""
import numpy as np

"""
# trajectories =  array of demonstration trajectories, each trajectory is an array of state action pairs
# Trajectories must all be same length?
# TODO: feature_matrix = array of feature vectors, each feature vector is associated with the state at that index
# transition_probabilities = probability of moving from one state to another given action, 1 for legal moves, 0 for illegal moves
# TODO: feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?
# policy = an array of length n_states containing which action to perform in corresponding state?
# states = all possible states in environment

N_ACTIONS = 4
N_STATES = # TODO
N_FEATURES = # TODO
LEARNING_RATE = 0.01
N_EPOCHS = 200
DISCOUNT = 0.01
THRESHOLD = 1e-2

def featureComputer(states, states_to_boards, features):
    feature_matrix = []
    for each state:
        board = states_to_boards[state]
        state_features = np.concat([i.process(board) for i in features])
        feature_matrix.append(state_features)

    feature_matrix = np.stack(feature_matrix)
    return feature_matrix

environment = gridworld_env
feature_vectors = getFeatureVectors()
feature_matrix = getFeatureMatrix(states, feature_vectors)
states = getStatesFromEnv()
transition_probabilities = getTransitionProbabilities(environment)
trajectories = getTrajectories()

maxEntIRL(states, feature_matrix, transition_probabilities, trajectories)

"""

def maxEntIRL(states, feature_matrix, transition_probabilities, trajectories,
              learning_rate=1e-2, n_epochs=1000):
    """Computes the weights for the features used in the construction of
    feature_matrix using maximum entropy IRL. The gradient step for the weights
    \theta is given by the loss L:

    \grad_{\theta} L = \alpha(empirical feature counts - feature counts at current \theta)

    with learning rate \alpha. Feature counts at current \theta are found by
    solving the MDP with rewards given by the current weights. This requires
    solving an MDP at every gradient step.

    Args:
        states: array of size (n_states)
        feature_matrix: array of size (n_states, n_features)
        transition_probabilities: array of size (n_states, n_actions, n_states)
        trajectories: array of size (n_trajectories, traj_length, 2)
        learning_rate: float
        n_epochs: int

    Returns:
        rewards: array of size (n_states)
        weights: array of size (n_features)
    """
    ## Initialisation
    n_states, n_features = feature_matrix.shape
    _, n_actions, _ = transition_probabilities.shape
    weights = numpy.random.uniform(size=(n_features))

    ## Get feature expectations
    feature_expectations = getFeatureExpectations(feature_matrix, trajectories)

    ## Gradient steps
    for i in range(n_epochs):
        rewards = feature_matrix.dot(weights)
        expected_svf = getExpectedSVF(rewards, transition_probabilities, trajectories)

        ## Should this be weights rather than rewards?
        rewards += learning_rate * (feature_expectations - feature_matrix.T.dot(expected_svf))

    ## Return rewards and weights
    return feature_matrix.dot(weights).reshape((n_states,)), weights


def getFeatureVectors(states?):

    # TODO
    return feature_vectors


def getStatesFromEnv():

    # TODO

    return states


def stateToInt(state):
    # TODO
    # Return int


def getTrajectories():

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


def getExpectedSVF(rewards, transition_probabilities, trajectories):
    """Computes the expected state visitation frequency vector for a given set
    of rewards by evaluating the policy and then using this to determine state
    occupancy probabilities at a given time. These are then summed over time.

    Reference code:
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
    """
    # expected state visitation frequencies
    policy = getPolicy(transition_probability, rewards)

    ## Initialisation
    n_states, n_actions, _ = transition_probabilities.shape
    num_traj, traj_length = trajectories.shape
    expected_svf = np.zeros((n_states, traj_length))

    ## Get initial state frequencies
    for trajectory in trajectories:
        ## second index to trajectory indicates using state, not action
        expected_svf[trajectory[0, 0]] += 1./num_traj # freq, not count

    # (I guess initial_state_probabilities would look like this: [0,1,0,0,0,0...] because our agent always starts in same place? - BUT in dynamic envs with more objects could not be.

    ## I suspect there's a more efficient way to do this
    for t in range(1, traj_length):
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] * policy[i,j] *
                                   transition_probabilities[i,j,k])

    # Sum over time and return
    return expected_svf.sum(axis=1)

def getPolicy(transition_probabilities, rewards, value_function=None):
    """Computes the optimal policy for a given transition probability and reward
    specification by first computing the value function and then taking greedy
    actions based on this.

    Reference code:
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/value_iteration.py
    """
    # Get optimal policy
    if value_function is None:
        value_function = getOptimalValueFunction(transition_probabilities, rewards, DISCOUNT, THRESHOLD)

    # If stochastic... do a thing here (see https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/value_iteration.py

    n_states, n_actions, _ = transition_probabilities.shape
    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (rewards[k] + discount_factor * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy


def getOptimalValueFunction(transition_probabilities, rewards, discount_factor,
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

        Q = rewards + gamma * np.dot(transition_probabilities, V_prev)
        V = np.amax(Q, axis=1)

        diff = np.amax(abs(V_prev - V))

    return V
