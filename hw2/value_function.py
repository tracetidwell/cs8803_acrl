import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from keras import models
from keras import layers
import keras.backend as K
from keras.utils.np_utils import to_categorical
import tensorflow as tf


class ValueFunction:
    
    def __init__(self, session, input_dim, scaler=StandardScaler):
        self.session = session
        self.input_dim = input_dim
        # self.gamma = gamma
        # self.scaler = scaler()
        # self.n_episodes = 0
        # self.states = []
        # self.rewards = []
        # self.episode_states = []
        # self.episode_rewards = []
                
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.fc1 = layers.Dense(16, activation='relu')(self.inputs)    
        self.fc2 = layers.Dense(16, activation='relu')(self.fc1)
        # self.fc3 = layers.Dense(64, activation='softmax')(self.fc2)
        # self.fc4 = layers.Dense(128, activation='softmax')(self.fc3)
        # self.fc5 = layers.Dense(256, activation='softmax')(self.fc4)
        self.output = layers.Dense(1, activation=None)(self.fc2)

        self.targets = tf.placeholder(tf.float32, shape=(None,), name='targets')
        
        loss = tf.reduce_sum(tf.square(self.targets - self.output))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})

    def train(self, states, returns):
        self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                   self.targets: returns})
    
    # def train(self, states, target_rewards):
    #     for i in range(self.n_episodes):
    #         discounted_rewards = np.array(self.rewards[i])
    #         for j in range(len(discounted_rewards)-2, -1, -1):
    #             discounted_rewards[i] += 0.9 * discounted_rewards[i+1]
    #         self.session.run(self.train_op, feed_dict={self.inputs: self.scaler.fit_transform(self.states[i]),
    #                                                    self.labels: discounted_rewards})

        # self.states = []
        # self.rewards = []
        # self.n_episodes = 0

    # def log_state_reward(self, state, reward):
    #     self.episode_states.append(state)
    #     self.episode_rewards.append(reward)

    # def log_episode(self):
    #     self.states.append(self.episode_states)
    #     self.rewards.append(self.episode_rewards)
    #     self.episode_states = []
    #     self.episode_rewards = []
    #     self.n_episodes += 1
