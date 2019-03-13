import numpy as np
from sklearn.preprocessing import OneHotEncoder

from keras import models
from keras import layers
import keras.backend as K
from keras.utils.np_utils import to_categorical
import tensorflow as tf


class Agent:
    
    def __init__(self, session, input_dim, state_encoder, policy, value_function, gamma=0.99):#, q_function):
        self.session = session
        self.input_dim = input_dim
        #self.buffer_size = buffer_size
        self.state_encoder = state_encoder
        self.policy = policy
        self.value_function = value_function
        #self.q_function = q_function
        self.gamma = gamma
        #self.buffer = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_rewards = []
        self.next_states = []
        self.dones = []
        self.advantages = []
        
    def add_experience(self, s, a, r, next_s, d):
        # if len(self.states) >= self.buffer_size:
        #     self.states.pop(0)
        #     self.actions.pop(0)
        #     self.rewards.pop(0)
        #     self.next_states.pop(0)
        #     self.dones.pop(0)
            
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(next_s)
        self.dones.append(d)
        
    def train(self):#, batch_size=128):
#         batch_size = min(batch_size, len(self.states))
#         indexes = np.random.choice(len(self.states), batch_size)
#         states = np.array(self.states)[indexes]
#         actions = np.array(self.actions)[indexes]
#         rewards = np.array(self.rewards)[indexes]
#         next_states = np.array(self.next_states)[indexes]
#         dones = np.array(self.dones)[indexes]
        
        self.calc_advantages_discounted_rewards()
        self.states = np.array(self.states).reshape(-1, self.input_dim)
        self.policy.train(np.array(self.states), np.array(self.actions), np.array(self.advantages))
        #self.q_function.train(states, actions, rewards, next_states, dones)
        self.value_function.train(self.states, self.discounted_rewards)
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_rewards = []
        self.advantages = []
        
    def get_legal_moves(self, game, piece=None):
        if piece:
            return [list(move) for move in list(game.legalMoves(piece))]
        else:
            return [list(move) for move in list(game.legalMoves())]
        
    def calc_advantages_discounted_rewards(self):
        #discounted_rewards = []
        #advantages = []
        G = 0
        for state, reward in zip(reversed(self.states), reversed(self.rewards)):
            self.discounted_rewards.append(G)
            self.advantages.append(G - self.value_function.predict(state).reshape(-1,))
            G = reward + self.gamma * G
        self.discounted_rewards.reverse()
        self.advantages.reverse()
        self.advantages = np.array(self.advantages).reshape(-1,)


class PolicyGradient:

    def __init__(self, session, input_dim, output_dim, learning_rate=0.1, gamma=0.99, gamma_decay=0.99, epsilon=1, epsilon_decay=0.001):
        self.session = session
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.actions_ = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        
        self.fc1 = layers.Dense(64, activation='relu')(self.inputs)
        self.fc2 = layers.Dense(64, activation='relu')(self.fc1)
        self.output = layers.Dense(self.output_dim, activation='softmax')(self.fc2)
        
        action_probs = tf.log(tf.reduce_sum(self.output * tf.one_hot(self.actions_, self.output_dim), reduction_indices=[1]))
        loss = -tf.reduce_sum(self.advantages - action_probs)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
    
    def train(self, states, actions, advantages):                
        self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                   self.actions_: actions,
                                                   self.advantages: advantages})
        
    def choose_action(self, moves, state):
        probs = self.predict(state).reshape(-1,)
        probs = probs[:len(moves)]
        print(probs)
        print(sum(probs))
        print(len(moves))
        if sum(probs) == 0:
            return np.random.choice(len(moves))    
        else:
            probs /= np.sum(probs)
            return np.random.choice(len(moves), p=probs)


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


class ValueFunction:
    
    def __init__(self, session, input_dim):
        self.session = session
        self.input_dim = input_dim
                
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=(None,), name='labels')
        
        self.fc1 = layers.Dense(64, activation='relu')(self.inputs)
        self.fc2 = layers.Dense(64, activation='relu')(self.fc1)
        self.output = layers.Dense(1, activation=None)(self.fc2)
        
        loss = tf.reduce_sum(tf.square(self.labels - self.output))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
    
    def train(self, states, rewards):              
        self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                   self.labels: rewards})


class StateEncoder:
    
    def __init__(self, n_pieces, depth):
        self.n_pieces = n_pieces
        self.depth = depth
        self.piece_encoder = OneHotEncoder(categories='auto')
        self.piece_encoder.fit(np.array(range(self.n_pieces)).reshape(-1, 1))
        
    def encode_piece(self, piece):
        #return self.piece_encoder.transform(np.array([piece]).reshape(-1, 1)).todense().fatten()
        return np.array(self.piece_encoder.transform(np.array([piece]).reshape(-1, 1)).todense()).flatten()
    
    def encode_board(self, top):
        diff = top - max(top)
        diff[diff<-self.depth] = -self.depth
        diff = diff / self.depth
        return diff
    
    def encode_state(self, game):
        top = np.array(list(game.getTop()))
        height = max(top)
        board = self.encode_board(top)
        piece = self.encode_piece(game.getNextPiece())
        return np.concatenate([board, [height/game.getRows()], piece]).reshape(1, -1)