from __future__ import print_function
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple
import datetime


# %% StateProcessor
class StateProcessor():
    """
    Changes gridworld RGB frames to gray scale.
    """
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[x_size, y_size, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [self.x_size, self.y_size, 3] gridworld RGB State

        Returns:
            A processed [self.x_size, self.y_size] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


# %% Estimator
class Estimator():
    """
    Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, actions_num, x_size, y_size, frames_state=2,
                 scope="estimator"):
        self.scope = scope
        self.actions_num = actions_num
        self.x_size = x_size
        self.y_size = y_size
        self.frames_state = frames_state
        # Writes Tensorboard summaries to disk
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are FRAMES_STATE RGB frames of shape of the gridworld
        self.X_pl = tf.placeholder(shape=[None, self.x_size, self.y_size,
                                          self.frames_state]
                                   , dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # NETWORK ARCHITECTURE
        # tf.contrib.layers.conv2d(input, num_outputs, kernel_size, stride)
        conv1 = tf.contrib.layers.conv2d(X, 64, 2, 1, activation_fn=tf.nn.relu)
        # try with padding = 'VALID'
        # pool1 = tf.contrib.layers.max_pool2d(conv1, 2)
        # conv2 = tf.contrib.layers.conv2d(pool1, 32, WX, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv1)
        fc1 = tf.contrib.layers.fully_connected(flattened, 64)
        self.predictions = tf.contrib.layers.fully_connected(fc1, self.actions_num)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, FRAMES_STATE, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, actions_num] containing the estimated
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, FRAMES_STATE, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        global_step, _, loss = sess.run(
                        [tf.train.get_global_step(), self.train_op, self.loss],
                        feed_dict)
        return loss


# %% helper functions
def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# %% DQNAgent
class DQNAgent():
    """
    DQNAgent adjusted to ai-safety-gridworlds.
    """
    def __init__(self,
                 sess,
                 world_shape,
                 actions_num,
                 env,
                 frames_state=2,
                 experiment_dir=None,
                 replay_memory_size=1500,
                 replay_memory_init_size=500,
                 update_target_estimator_every=250,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=50000,
                 batch_size=32):

        self.world_shape = world_shape
        self.actions_num = actions_num
        self.frames_state = frames_state
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.q = Estimator(actions_num, world_shape[0], world_shape[1], frames_state=frames_state, scope="q")
        self.target_q = Estimator(actions_num, world_shape[0], world_shape[1], frames_state=frames_state, scope="target_q")
        self.sp = StateProcessor(world_shape[0], world_shape[1])
        self.replay_memory = []
        self.replay_memory_size = replay_memory_size
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.policy = make_epsilon_greedy_policy(self.q, actions_num)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.total_t = 0

        self.saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        self.experiment_dir = experiment_dir
        if experiment_dir:
            self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            self.checkpoint_path = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)

        self.new_episode()
        time_step = env.reset()
        state = self.get_state(time_step.observation)
        for i in range(replay_memory_init_size):
            action = self.act(time_step.observation, eps=0.95)
            time_step = env.step(action)
            next_state = self.get_state(time_step.observation)
            done = time_step.last()

            assert state is not None
            assert next_state is not None
            self.replay_memory.append(Transition(state, action, time_step.reward, next_state, done))
            if done:
                time_step = env.reset()
                self.new_episode()
                state = self.get_state(time_step.observation)
            else:
                state = next_state

    def new_episode(self):
        self.loss = None
        self.prev_state = None

    def get_state(self, obs):
        frame = np.moveaxis(obs['RGB'], 0, -1)
        frame = self.sp.process(self.sess, frame)
        if self.prev_state is None:
            state = np.stack([frame] * self.frames_state, axis=2)
        else:
            state = np.stack([self.prev_state[:,:,self.frames_state - 1], frame], axis=2)
        return state

    def act(self, obs, eps=None):
        if eps is None:
            eps = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        state = self.get_state(obs)
        probs = self.policy(self.sess, state, eps)  # you want some very random experience to populate the replay memory
        self.prev_state = state
        return np.random.choice(self.actions_num, p=probs)

    def learn(self, time_step, action):

        if self.total_t % self.update_target_estimator_every == 0:
            copy_model_parameters(self.sess, self.q, self.target_q)

        next_state = self.get_state(time_step.observation)
        done = time_step.last()
        if len(self.replay_memory) == self.replay_memory_size:
            self.replay_memory.pop(0)

        self.replay_memory.append(Transition(self.prev_state, action,
                                             time_step.reward, next_state, done))

        # finally! let's learn something:
        sample = np.random.choice(len(self.replay_memory), self.batch_size)
        sample = [self.replay_memory[i] for i in sample]

        sts, a, r, n_sts, d = tuple(map(np.array, zip(*sample)))
        qs = self.target_q.predict(self.sess, n_sts).max(axis=1)
        qs[d] = 0
        targets = r + self.discount_factor * qs
        loss = self.q.update(self.sess, sts, a, targets)

        self.total_t += 1
        if time_step.last():
            self.new_episode()
        return loss

    def save(self):
        if self.experiment_dir:
            self.saver.save(tf.get_default_session(), self.checkpoint_path)
