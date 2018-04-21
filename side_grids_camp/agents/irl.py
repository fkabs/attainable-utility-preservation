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


"""
# trajectories =  array of demonstration trajectories, each trajectory is an array of state action pairs
# Trajectories must all be same length?
# feature_matrix = array of feature vectors, each feature vector is associated with the state at that index
# transition_probabilities = probability of moving from one state to another given action, 1 for legal moves, 0 for illegal moves
# feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?
# policy = an array of length n_states containing which action to perform in corresponding state?
# states = all possible states in environment
"""

import numpy as np
from ai_safety_gridworlds.demonstrations import demonstrations
import matplotlib.pyplot as plt
from itertools import product

def trajectory_from_demo(demo, env, board_mapper, board_state_map):
    """Maps a Demonstration object (from SafetyGame) to a trajectory of shape
    [len(demo.actions), 2] comprising (state, action) pairs for every timestep.

    The random seed is set for each demonstration so stochastic environments
    behave correctly.

    Args:
        demo: a SafetyGame Demonstration object
        env: a pycolab environment
        board_mapper: a dictionary mapping boards to state indices

    Returns:
        trajectory: array of size (len_traj, 2)
    """
    np.random.seed(demo.seed)
    env.reset()
    agent_gs = 2.
    box_gs = 4.

    time_step = env.reset()
    init_state_idx = board_mapper(time_step.observation['board'],
                                  board_state_map,
                                  agent_gs,
                                  box_gs)

    states = [init_state_idx,]
    actions = []

    for action in demo.actions:
        time_step = env.step(action)
        state_idx = board_mapper(time_step.observation['board'],
                                 board_state_map,
                                 agent_gs,
                                 box_gs)

        states.append(state_idx)
        actions.append(action.value)

    actions.append(action.QUIT) # end the trajectory
    trajectory = np.stack((states, actions))
    return trajectory

def make_trajectories(demos, env, board_mapper, board_state_map):
    trajectories = [trajectory_from_demo(demo, env, board_mapper, board_state_map) for demo in demos]
    return np.array(trajectories)

def maxEntIRL(states, feature_matrix, transition_probabilities, trajectories,
              learning_rate=1e-2, n_epochs=1000, horizon=100, discount=1,
              weight_init=None):
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
    
    if weight_init is None:
        weights = -1.*np.random.uniform(size=(n_features))

    else:
        weights = weight_init

    ## Get feature expectations
    feature_expectations = getFeatureExpectations(feature_matrix, trajectories)

    ## Gradient steps
    for i in range(n_epochs):
        rewards = feature_matrix.dot(weights)

        expected_svf = getExpectedSVF(rewards,
                                      transition_probabilities,
                                      trajectories,
                                      horizon=horizon,
                                      discount=discount)

        ## Should this be weights rather than rewards?
        weights += learning_rate * (feature_expectations - feature_matrix.T.dot(expected_svf))

    ## Return rewards and weights
    return feature_matrix.dot(weights).reshape((n_states,)), weights

def getFeatureExpectations(feature_matrix, trajectories):

    # feature_expectations = sum feature vectors of every state visited in every demo trajectory, divide result by the number of trajectories, to get feature expectation (over all states) for an average demo trajectory?

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations


def getExpectedSVF(rewards, transition_probabilities, trajectories, horizon,
                   discount=1):
    """Computes the expected state visitation frequency vector for a given set
    of rewards by evaluating the policy and then using this to determine state
    occupancy probabilities at a given time. These are then summed over time.

    Reference code:
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
    """
    # expected state visitation frequencies
    policy = getPolicy(transition_probabilities,
                       rewards,
                       horizon=horizon,
                       discount=discount)

    ## Initialisation
    n_states, n_actions, _ = transition_probabilities.shape
    num_traj, traj_length, _ = trajectories.shape
    expected_svf = np.zeros((n_states, traj_length))

    ## Get initial state frequencies
    for trajectory in trajectories:
        ## second index to trajectory indicates using state, not action
        expected_svf[trajectory[0, 0], 0] += 1./num_traj # freq, not count

    # (I guess initial_state_probabilities would look like this: [0,1,0,0,0,0...] because our agent always starts in same place? - BUT in dynamic envs with more objects could not be.

    ## I suspect there's a more efficient way to do this
    for t in range(1, traj_length):
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] * policy[i,j] *
                                   transition_probabilities[i,j,k])

    # Sum over time and return
    return expected_svf.sum(axis=1)

def getPolicy(transition_probabilities, rewards, discount=1,
              threshold=1e-4, value_function=None, horizon=100):
    """Computes the optimal policy for a given transition probability and reward
    specification by first computing the value function and then taking greedy
    actions based on this.

    Reference code:
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/value_iteration.py
    """
    # Get optimal policy
    if value_function is None:
        value_function = getOptimalValueFunction(transition_probabilities,
                                                 rewards,
                                                 discount,
                                                 threshold,
                                                 horizon)

    # If stochastic... do a thing here (see https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/value_iteration.py

    n_states, n_actions, _ = transition_probabilities.shape
    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (rewards[k] + discount * value_function[k])
                                     for k in range(n_states)))
    policy_indices = np.array([_policy(s) for s in range(n_states)])

    ## Should incorporate tie-breaking better
    policy = np.zeros((n_states, n_actions))
    for s, idx in enumerate(policy_indices):
        policy[s, idx] = 1
    return policy


def getOptimalValueFunction(transition_probabilities, rewards, discount,
                            conv_threshold, horizon=100):
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
        discount: float in [0,1]
        conv_threshold: float setting convergence threshold
        horizon: number of timesteps to evaluate out to

    Returns a vector of values of length n_states. The following code was used
    as a reference:
    https://github.com/krasheninnikov/max-causal-ent-irl/blob/master/value_iter_and_policy.py
    https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
    """

    n_states, n_actions, _ = transition_probabilities.shape
    V = np.copy(rewards) # initialise value at rewards
    Q = np.zeros((n_states, n_actions))

    t = 0
    diff = float("inf")
    while diff > conv_threshold:
        V_prev = np.copy(V)

        for s in range(n_states):
            for a in range(n_actions):
                ## Bellman update; goal states are sinks
                s_prime = np.argmax(transition_probabilities[s,a,:])
                Q[s,a] = rewards[s_prime] + discount * V_prev[s_prime]

        V = np.amax(Q, axis=1)
        diff = np.amax(abs(V_prev-V))

        t += 1
        if horizon is not None:
            if t==horizon: break

    return V
