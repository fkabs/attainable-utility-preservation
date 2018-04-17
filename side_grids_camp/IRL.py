#Thanks to:
@misc{alger16,
  author       = {Matthew Alger},
  title        = {Inverse Reinforcement Learning},
  year         = 2016,
  doi          = {10.5281/zenodo.555999},
  url          = {https://doi.org/10.5281/zenodo.555999}
}

import numpy as np


# Instantiate gridworld

# Each state represented by integer? and associated feature vector

# array of trajectories, each trajectory is an array of state action pairs
trajectories = getExampleTrajectories()

# array of feature vectors, each feature vector is associated with the state at that index
feature_matrix = getFeatureMatrix()

# probability of moving from one state to another given action, 1 for legal moves, 0 for illegal moves
trans_probs = getTransitionProbabilities()


num_of_actions = 4

irl = maxEntIRL(feature_matrix, num_of_actions, discount, trans_probs, trajectories, epochs, learning_rate)


# TODO methods to map from state to state index and back
# How are states represented here - location of agent and box? Hard to apply in complex envs. Feature array? Greyscale images?

def stateToInt(state):
    # TODO
    # Return int
    


def getExampleTrajectories():
    
    trajectories = []
    # TODO
    # (each state in traj is tuple (state, action, reward obtained)
    # Could generate (take num, len, policy)or do by hand here   
    return trajectories
    
def getFeatureMatrix(all_states, feature_vectors):
    
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
    
    # the average feature vector?
    
    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations


def getSVF(number_of_states, trajectories):
    
    svf = np.zeros(n_states)

        for trajectory in trajectories:
            for state, _, _ in trajectory:
                svf[state] += 1

        svf /= trajectories.shape[0]

    return svf


def getExpectedSVF(number_of_states, rewards, number_of_actions, discount, transition_probability, trajectories):
    # state visitation frequency
    # policy = findPolicy(n_states, n_actions, transition_probability, rewards, discount)
    
    # 
    
    return expected_svf
    
    

def findPolicy(n_states, n_actions, transition_probability, rewards, discount):
    
    # TODO
    


def maxEntIRL(feature_matrix, num_of_actions, discount, trans_probs, trajectories, epochs, learning_rate):
    
    # TODO
    
    # Initialise weights = numpy.random.uniform(size=(num_of_states,))
    # Get feature matrix using all states and feature vectors
    
    # for i in range epochs
        # rewards = feature_matrix.dot(weights)
        # get expected svfs (state visitation frequencies)
        # rewards += learning_rate * (feature_expectations - feature_matrix * expected svfs)
        
    # return feature_matrix* rewards
    
        
    
    
    
    return rewards


    