import numpy as np
import random
from collections import namedtuple, deque
from numpy_ringbuffer import RingBuffer

from tensorflow.python.framework import dtypes

from nn_helpers.dqn.model import QNetworkTf

import tensorflow as tf

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, network, state_size, action_size, seed=42, save=True, save_loc='models/model.ckpt', save_every=100):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.sess = tf.Session()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = network(
            self.sess, state_size, action_size, "local")
        self.qnetwork_target = network(
            self.sess, state_size, action_size, "target")

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

        self.soft_update_op = self._get_soft_update_op()

        self.loss = 0
        self.learning_steps = 0

        self.eps = 0.0
        self.eps_end = 0.0
        self.eps_decay = 0.0

        self.saver = tf.train.Saver()
        self.save_config = {'save': save,
                            'save_loc': save_loc, 'save_every': save_every}

        self.episode_counter = 0

        self.act_op = tf.cond(tf.random_uniform([1], dtype=tf.float32)[0] > self.eps,
                         lambda: tf.argmax(self.qnetwork_local.output, output_type=tf.int32, axis=1),
                         lambda: tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32))
    
    def set_epsilon(self, epsilon_start, epsilon_end, episodes_till_cap):
        self.eps = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_end**(1/episodes_till_cap)

    def reset_episode(self):
        self.episode_counter += 1
        self.eps = max(self.eps_end, self.eps_decay*self.eps)

        if self.save_config['save'] and self.episode_counter % self.save_config['save_every'] == 0:
            self.saver.save(self.sess, "models/model.ckpt")

    def _get_soft_update_op(self):
        Qvars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference_local')
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference_target')

        return [tvar.assign(TAU*qvar + (1.0-TAU)*tvar) for qvar, tvar in zip(Qvars, target_vars)]

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, mode="train"):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            mode (str): "train" or "test", for strategy choosing
        """
        return self.sess.run(self.act_op, feed_dict={self.qnetwork_local.input: [state]})

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        q_target_output = self.qnetwork_target.forward(next_states)
        Q_targets_next = np.expand_dims(
            np.amax(q_target_output, 1), 1)
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))

        loss, result = self.qnetwork_local.train(states, Q_targets, actions)
        self._update_loss(loss)
        self.memory.set_priority(experiences, abs(np.max(q_target_output, axis=1) - np.max(result, axis=1)))

        self.soft_update()

    def soft_update(self):
        self.sess.run(self.soft_update_op)

    def _update_loss(self, loss):
        self.loss = self.loss*self.learning_steps / \
            (self.learning_steps+1) + loss/(self.learning_steps+1)
        self.learning_steps += 1


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.filled = False

        self.pointer = 0
        # self.memory = deque(maxlen=buffer_size)
        self.memory = np.empty((buffer_size, 5), dtype=object)
        self.priorities = np.empty(buffer_size, dtype=np.float32)
        # self.memory = RingBuffer(buffer_size, dtype=object)
        # self.priorities = RingBuffer(buffer_size, dtype=np.float32)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.last_sample = None

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory[self.pointer] = e
        self.priorities[self.pointer] = 0.1
        # self.memory.append(e)
        # self.priorities.append(0.1)
        self.pointer = (self.pointer+1)%self.buffer_size
        if self.pointer==0:
            self.filled = True

    def set_priority(self, experiences, error):
        # indices = np.argwhere(self.memory==experiences)
        # self.priorities[indices] = error
        # for i, j in zip(experiences, error):
        #     self.priorities[np.argwhere(self.memory==i)] = j
        self.priorities[self.last_sample] = error

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if not self.filled:
            experiences = np.random.choice(np.arange(self.pointer), size=self.batch_size)
        else:
            experiences = np.random.choice(np.arange(self.buffer_size), size=self.batch_size)
        self.last_sample = experiences
        experiences = self.memory[experiences]
        # experiences = np.random.choice(self.memory, size=self.batch_size, p=np.array(self.priorities))

        # experiences = self.memory[experiences]
        
        states = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None])
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.vstack(
            [e[3] for e in experiences if e is not None])
        dones = np.vstack(
            [e[4] for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
