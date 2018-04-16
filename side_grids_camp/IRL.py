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

trajectories = getExampleTrajectories()
feature_matrix = getFeatureMatrix()
trans_probs = getTransitionProbabilities()
num_of_actions = 4

irl = maxEntIRL(feature_matrix, num_of_actions, discount, trans_probs, trajectories, epochs, learning_rate)


def getExampleTrajectories():
    
    trajectories = []
    # TODO
    # (each state in traj is tuple (state, action, reward obtained)
    # Could generate (take num, len, policy)or do by hand here   
    return trajectories
    
def getFeatureMatrix():
    
    feature_matrix = []
    # TODO 
    # Each state has associated feature vector  
    # for n in range (number of possible states)
    # append feature vector
    return feature_matrix


def getTransitionProbabilities():
    
    trans_probs = []
    # TODO
    # trans_probs is prob of transitions from state i to k given action j
    # so 3-loop for each state, action, state again
    # IF NOT WIND
    # For (agent_pos to agent_pos+1, box_pos to box_pos+1)
    # if not adjacent , or if impassable, or if box move but agent not next to it then 0
    # else 1 (- wind prob)
    # But this is agent only - what about dynamic envs?
    
    return trans_probs


def getFeatureExpectations(feature_matrix, trajectories):
    
    # Feature expectations are the average features of the trajectories
    
    # For each trajectory
    # for each state in trajectory

    return feature_expectations


def getStateVisitationFrequency(number_of_states, trajectories):
    
    svf = np.zeros(n_states)

    # For each trajectory
    # for each state in trajectory
        # Because each state is represented as an integer
        svf[state] += 1

    svf / trajectories.shape[0]

    return svf
    


def maxEntIRL(feature_matrix, num_of_actions, discount, trans_probs, trajectories, epochs, learning_rate):
    
    # TODO
    
    
    
    return 

    