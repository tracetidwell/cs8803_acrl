import numpy as np
from sklearn.preprocessing import OneHotEncoder

#from keras import models
#from keras import layers, initializers
#import keras.backend as K
#from keras.utils.np_utils import to_categorical
import tensorflow as tf


class Agent:
    
    def __init__(self, session, input_dim, output_dim, state_encoder, policy, value_function, gamma=0.9):#, q_function):
        self.session = session
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_encoder = state_encoder
        self.policy = policy
        self.value_function = value_function
        self.gamma = gamma
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_next_states = []
        self.episode_dones = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.n_episodes = 0

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())


    def encode_state(self, tetris):
        return self.state_encoder.transform(tetris)

        
    def log_experience(self, s, a, r, next_s, d):            
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)
        self.episode_next_states.append(next_s)
        self.episode_dones.append(d)


    def log_episode(self):
        self.states.append(self.episode_states)
        self.actions.append(self.episode_actions)
        self.rewards.append(self.episode_rewards)
        self.next_states.append(self.episode_next_states)
        self.dones.append(self.episode_dones)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_next_states = []
        self.episode_dones = []
        self.n_episodes += 1


    def reset_experience(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_next_states = []
        self.episode_dones = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


    def choose_best_action(self, tetris, state):
        moves = tetris.get_legal_moves()
        probs = self.policy.predict(state)[0]
        return int(np.argmax(probs[:len(moves)]))


    def choose_action(self, tetris, state):
        moves = tetris.get_legal_moves()
        probs = self.policy.predict(state)[0]
        if len(moves) != self.output_dim:
            probs = probs[:len(moves)]
            probs /= sum(probs)
        return np.random.choice(len(moves), p=probs)

        
    def train(self):
        for i in range(self.n_episodes):
            advantages, discounted_rewards = self.calc_advantages_discounted_rewards(self.states[i], self.rewards[i])
            self.policy.train(np.array(self.states[i]).reshape(-1, self.input_dim), self.actions[i], advantages)
            self.value_function.train(np.array(self.states[i]).reshape(-1, self.input_dim), discounted_rewards)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.n_episodes = 0
    
        
    def calc_advantages_discounted_rewards(self, states, rewards):
        G = 0
        discounted_rewards = []
        advantages = []
        for state, reward in zip(reversed(states), reversed(rewards)):
            discounted_rewards.append(G)
            advantages.append(G - self.value_function.predict(state).reshape(-1,))
            G = reward + self.gamma * G
        discounted_rewards.reverse()
        advantages.reverse()
        advantages = np.array(advantages).reshape(-1,)
        return advantages, discounted_rewards


class PolicyGradient:

    def __init__(self, session, input_dim, output_dim, learning_rate=0.01):
        self.session = session
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.actions_ = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        self.fc1 = tf.layers.dense(inputs=self.inputs, units=512, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.fc2 = tf.layers.dense(inputs=self.fc1, units=256, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.fc3 = tf.layers.dense(inputs=self.fc2, units=128, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.fc4 = tf.layers.dense(inputs=self.fc3, units=64, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.output = tf.layers.dense(inputs=self.fc4, units=self.output_dim, activation=tf.nn.softmax, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))

        # self.layer_1 = layers.Dense(512, bias_initializer=initializers.Constant(0.1), activation='relu')
        # self.layer_2 = layers.Dense(256, bias_initializer=initializers.Constant(0.1), activation='relu')
        # self.layer_3 = layers.Dense(128, bias_initializer=initializers.Constant(0.1), activation='relu')
        # self.layer_4 = layers.Dense(64, bias_initializer=initializers.Constant(0.1), activation='relu')
        # self.layer_5 = layers.Dense(self.output_dim, bias_initializer=initializers.Constant(0.1), activation='softmax')
        
        # self.fc1 = self.layer_1(self.inputs)
        # self.fc2 = self.layer_2(self.fc1)
        # self.fc3 = self.layer_3(self.fc2)
        # self.fc4 = self.layer_4(self.fc3)
        # self.output = self.layer_5(self.fc4)
        
        action_probs = tf.log(tf.reduce_sum(self.output * tf.one_hot(self.actions_, self.output_dim), reduction_indices=[1]))
        loss = -tf.reduce_sum(self.advantages * action_probs)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.outputs[0], labels=np.array(self.actions))
        # loss = tf.reduce_mean(neg_log_prob * self.discounted_rewards())
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
    
    def train(self, states, actions, advantages, n_epochs=5):
        for i in range(n_epochs):
            self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                       self.actions_: actions,
                                                       self.advantages: advantages})
        

class ValueFunction:
    
    def __init__(self, session, input_dim, learning_rate=0.01):
        self.session = session
        self.input_dim = input_dim
        self.learning_rate = learning_rate
                
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=(None,), name='labels')

        self.fc1 = tf.layers.dense(inputs=self.inputs, units=512, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.fc2 = tf.layers.dense(inputs=self.fc1, units=256, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.fc3 = tf.layers.dense(inputs=self.fc2, units=128, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.fc4 = tf.layers.dense(inputs=self.fc3, units=64, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        self.output = tf.layers.dense(inputs=self.fc4, units=1, activation=None, kernel_initializer=tf.glorot_normal_initializer, bias_initializer=tf.constant_initializer(0.1))
        
        # self.fc1 = layers.Dense(512, activation='relu')(self.inputs)
        # self.fc2 = layers.Dense(256, activation='relu')(self.fc1)
        # self.fc3 = layers.Dense(128, activation='relu')(self.fc2)
        # self.fc4 = layers.Dense(64, activation='relu')(self.fc3)
        # self.output = layers.Dense(1, activation=None)(self.fc4)
        
        loss = tf.reduce_sum(tf.square(self.labels - self.output))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
    
    def train(self, states, rewards, n_epochs=5):
        for i in range(n_epochs):
            self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                       self.labels: rewards})


class QFunction:
    
    def __init__(self, session, input_dim, output_dim, learning_rate=0.001, gamma=0.99, gamma_decay=0.99, epsilon=1, epsilon_decay=0.001):
        self.session = session
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=(None,), name='labels')
        self.actions_ = tf.placeholder(tf.int32, shape=(None,), name='actions')
        
        self.fc1 = layers.Dense(64, activation='relu')(self.inputs)
        self.fc2 = layers.Dense(64, activation='relu')(self.fc1)
        self.output = layers.Dense(self.output_dim, activation=None)(self.fc2)
        
        action_prob = tf.reduce_sum(self.output * tf.one_hot(self.actions_, self.output_dim))
        loss = tf.reduce_sum(tf.square(self.labels - action_prob))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
    
    def train(self, states, actions, rewards, next_states, dones):
        Qs_next_state = self.predict(next_states)
        target_Qs = []
        
        for i in range(len(states)):
            if dones[i]:
                target_Qs.append(rewards[i])
            else:
                target_Qs.append(rewards[i] * self.gamma * np.max(Qs_next_state[i]))
                
        self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                   self.labels: target_Qs,
                                                   self.actions_: actions})
        
    def choose_action(self, moves, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(moves))
        else:
            return np.argmax(self.predict(state))
        
    def best_value(self, state):
        return np.max(self.predict(state))




