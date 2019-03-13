#import matplotlib.pyplot as plt
from py4j.java_gateway import JavaGateway
import numpy as np
import struct
from sklearn.preprocessing import OneHotEncoder
#%matplotlib inline

from keras import callbacks
from keras import models
from keras import layers
import keras.backend as K
from keras.utils.np_utils import to_categorical
import tensorflow as tf

def piece_encoder(piece):
    piece_dict = {0: [0, 0, 0],
                    1: [0, 0, 1],
                    2: [0, 1, 0],
                    3: [0, 1, 1],
                    4: [1, 0, 0], 
                    5: [1, 0, 1], 
                    6: [1, 1, 0]}
    
    return piece_dict[piece]

def board_encoder(game):
    top = np.array(list(game.getTop()))
    diff = np.array(top) - max(top)
    diff[diff<-3] = -3
    diff /= 3
    return diff

class DQN:
    
    def __init__(self, input_dim, output_dim, buffer_size, sess, piece_encoder, board_encoder, 
                 learning_rate=0.001, gamma=0.99, gamma_decay=0.99, epsilon=1, epsilon_decay=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.buffer_size = buffer_size
        self.session = sess
        self.piece_encoder = piece_encoder
        self.board_encoder = board_encoder
        self.lr = learning_rate
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        #self.board_encoder = OneHotEncoder(categories='auto')
        #self.board_encoder.fit(np.array(range(3)).reshape(-1, 1))
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=(None,), name='labels')
        self.actions_ = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # self.fc1 = tf.layers.dense(inputs = self.inputs, 
        #                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                            units = 64, 
        #                            activation=tf.nn.relu)

        # self.fc2 = tf.layers.dense(inputs = self.fc1, 
        #                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                            units = 64, 
        #                            activation=tf.nn.relu)

        # self.output = tf.layers.dense(inputs = self.fc2, 
        #                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                               units = self.output_dim, 
        #                               activation=None)
        
        self.fc1 = layers.Dense(64, activation='relu')(self.inputs)
        self.fc2 = layers.Dense(64, activation='relu')(self.fc1)
        self.output = layers.Dense(self.output_dim, activation=None)(self.fc2)
        
        action_prob = tf.reduce_sum(self.output * tf.one_hot(self.actions_, self.output_dim))
        loss = tf.reduce_sum(tf.square(self.labels - action_prob))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

#         self.model = models.Sequential()
#         self.model.add(layers.Dense(64, activation='relu', input_shape=(self.input_dim,)))
#         self.model.add(layers.Dense(64, activation='relu'))
#         self.model.add(layers.Dense(self.output_dim, activation=None))
#         model.compile(optimizer='rmsprop', loss=self.pg_loss, metrics=['accuracy'])
        
#         self.X = tf.placeholder(tf.float32, shape=(None,), name='X')
#         y_pred = tf.convert_to_tensor(self.predict(self.X))
        
        #self.y_pred = tf.placeholder(tf.float32, shape=(None,), name='y_pred')
        
    def add_experience(self, s, a, r, next_s, d):
        if len(self.states) >= self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(next_s)
        self.dones.append(d)

    def state_encoder(self, game):
        top = list(game.getTop())
        diffs = max(top) - np.array(top)
        diffs[diffs > 1] = 2
        encoded_board = self.board_encoder(game)
        encoded_piece = self.piece_encoder(game.getNextPiece())
        return np.concatenate([encoded_board, encoded_piece])
        #return np.concatenate([diffs, encoded_piece])
        
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
        #return self.model.predict(X)
    
    def train(self, batch_size=128):
        batch_size = min(batch_size, len(self.states))
        indexes = np.random.choice(len(self.states), batch_size)
        states = np.array(self.states)[indexes]
        actions = np.array(self.actions)[indexes]
        rewards = np.array(self.rewards)[indexes]
        next_states = np.array(self.next_states)[indexes]
        dones = np.array(self.dones)[indexes]
        Qs_next_state = self.predict(next_states)
        target_Qs = []
        
        for i in range(batch_size):
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
    
    def get_legal_moves(self, game, piece=None):
        if piece:
            return [list(move) for move in list(game.legalMoves(piece))]
        else:
            return [list(move) for move in list(game.legalMoves())]
        
        
#     def dqn_loss(self, y_true, y_pred):
#         action_prob = tf.reduce_sum(y_pred * tf.one_hot([10, 14], 34))
#         neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.outputs[0], labels=np.array(self.actions))
#         return tf.reduce_mean(neg_log_prob * self.discounted_rewards())
        
#     def build_model(self):
#         model = models.Sequential()
#         model.add(layers.Dense(64, activation='relu', input_shape=(self.input_dim,)))
#         model.add(layers.Dense(64, activation='relu'))
#         model.add(layers.Dense(self.output_dim, activation=None))
#         model.compile(optimizer='rmsprop', loss=self.pg_loss, metrics=['accuracy'])
#         #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#         self.model = model


class Boltzmann_distribution:
    
    def __init__(self, input_dim, output_dim, mean=0, std=0.1, theta=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not theta:
            self.theta = np.random.uniform(mean, std, (1, input_dim))
            
    def get_legal_moves(self, game, piece=None):
        if piece:
            return [list(move) for move in list(game.legalMoves(piece))]
        else:
            return [list(move) for move in list(game.legalMoves())]
            
    def get_action_gradient(self, game, state_encoder):
        moves = self.get_legal_moves(game)
        action_states = np.array([state_encoder(list(game.projectMove(orient, slot))) for orient, slot in moves]).T
        states = np.concatenate([action_states, np.zeros((self.input_dim, self.output_dim-len(moves)))], axis=1)
        probs = np.exp(np.matmul(self.theta, states))
        probs[probs==1] = 0
        probs /= np.sum(probs)
        action = np.random.choice(self.output_dim, p=probs.flatten())
        gradient = states[:, action] - np.sum(states * probs, axis=1)
        
#         probs = np.zeros(len(moves))
#         expectation = np.zeros(self.input_dim)
#         for i, (orient, slot) in enumerate(moves):
#             state = state_encoder(list(game.projectMove(orient, slot)))
#             probs[i] = np.matmul(self.theta, state)
#             expectation += state * probs[i]
#         probs /= sum(probs)
#         action = np.random.choice(len(moves), p=probs)
#         gradient = state_encoder(list(game.projectMove(action))) - expectation
        return action, gradient
    
    def get_gradient(self, game, action, state_encoder):
        moves = self.get_legal_moves(game)
        probs = self.get_actions(game, state_encoder)
        expectation = np.zeros(self.input_dim)
        for i, (orient, slot) in enumerate(moves):
            expectation += state_encoder(list(game.projectMove(orient, slot))) * probs[i]
        return state_encoder(list(game.projectMove(action))) - expectation
    
    def update_theta(self, del_theta):
        self.theta += del_theta


class Agent:
    
    def __init__(self, state_values, input_dim, output_dim, policy=Boltzmann_distribution, alpha=0.001):
        self.trajectory = []
        self.trajectories = []
        self.reward = []
        self.rewards = []
        self.gradient = []
        self.gradients = []
        self.n_episodes = 0
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.policy = policy(input_dim, output_dim)
        self.board_encoder = OneHotEncoder(categories='auto')
        self.board_encoder.fit(np.array(range(state_values)).reshape(-1, 1))
        #self.piece_encoder = OneHotEncoder(categories='auto')
        #self.piece_encoder.fit(np.array(range(game.getPieces())).reshape(-1, 1))
        self.piece_encoder = {0: [0, 0, 0],
                              1: [0, 0, 1],
                              2: [0, 1, 0],
                              3: [0, 1, 1],
                              4: [1, 0, 0], 
                              5: [1, 0, 1], 
                              6: [1, 1, 0]}
        
    def state_encoder(self, top):
        #field = np.frombuffer(game.getByteArray(), dtype=np.int32)
        #heights = np.argmax(field.reshape(21, 10), axis=0)
        diffs = max(top) - np.array(top)
        diffs[diffs > 1] = 2
        #encoded_board = np.array(self.board_encoder.transform(diffs.reshape(-1, 1)).todense()).flatten()
        encoded_piece = self.piece_encoder[tetris.getNextPiece()]
        #return np.concatenate([encoded_board, encoded_piece])
        return np.concatenate([diffs, encoded_piece])
    
    def log_sar(self, s, a, r):
        self.trajectory.append((s, a))
        self.reward.append(r)
        #self.gradient.append(g)
        
    def log_episode(self):
        self.trajectories.append(self.trajectory)
        self.rewards.append(self.reward)
        self.gradients.append(self.gradient)
        self.n_episodes += 1
        self.trajectory = []
        self.reward = []
        self.gradient = []
    
    def get_legal_moves(self, game, piece=None):
        if piece:
            return [list(move) for move in list(game.legalMoves(piece))]
        else:
            return [list(move) for move in list(game.legalMoves())]
    
    def choose_action(self, game):
        action, gradient = self.policy.get_action_gradient(game, self.state_encoder)
        self.gradient.append(gradient)
        return action
        #return np.random.choice(len(actions), p=actions)
    
    def calc_gradient(self, game, action):
        return self.policy.get_gradient(game, action, self.state_encoder)
    
    def update_policy(self):
        total_gradient = np.zeros(self.input_dim)
        for i in range(self.n_episodes):
            for j, gradient in enumerate(self.gradients[i]):
                total_gradient += np.array(gradient) * sum(self.rewards[i][j:])
        total_gradient /= self.n_episodes
        self.policy.update_theta(self.alpha * total_gradient)
        self.trajectories = []
        self.rewards = []
        self.gradients = []
        self.n_episodes = 0
        
#         total_gradient = np.zeros(self.input_dim)
#         for i in range(self.n_episodes):
#             total_gradient += np.array(self.gradients[i]).sum(axis=0) * sum(self.rewards[i])
#         total_gradient /= self.n_episodes
#         self.policy.update_theta(self.alpha * total_gradient)
#         self.trajectories = []
#         self.rewards = []
#         self.gradients = []
#         self.n_episodes = 0




early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

class PolicyGradient:
    
    def __init__(self, input_dim, output_dim, learning_rate=0.01, gamma=0.99, decay_rate=0.99):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.states = []
        self.actions = []
        self.rewards = []
        self.build_model()
        
    def pg_loss(self, y_true, y_pred):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.outputs[0], labels=np.array(self.actions))
        return tf.reduce_mean(neg_log_prob * self.discounted_rewards())
        
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.input_dim,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.output_dim, activation=None))
        model.compile(optimizer='rmsprop', loss=self.pg_loss, metrics=['accuracy'])
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        
    def log_sar(self, s, a, r):
        self.states.append(s)
        actions = np.zeros(self.output_dim)
        actions[a] = 1
        self.actions.append(actions)
        self.rewards.append(r)
        
    def choose_action(self, state, legal_moves):
        n_moves = len(legal_moves)
        probs = K.eval(K.sigmoid(self.model.predict(state[None, :])))
        return np.random.choice(n_moves, p=(probs.flatten()[:n_moves] / sum(probs.flatten()[:n_moves])))
        #return np.random.choice(len(probs.flatten()), p=probs.flatten())
    
    def discounted_rewards(self):
        discounted_rewards = np.zeros(len(self.rewards))
        total_rewards = 0
        for t in reversed(range(len(self.rewards))):
            total_rewards = total_rewards * self.gamma + self.rewards[t]
            discounted_rewards[t] = total_rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) 
        return discounted_rewards
    
    def train(self):
        #split = int(np.array(self.states).shape[0]*0.2)
        #x_train = np.array(self.states)[split:]
        #y_train = np.array(self.actions)[split:]
        #x_val = np.array(self.states)[:split]
        #y_val = np.array(self.actions)[:split]
        #model.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=False)
        model.fit(np.array(self.states), np.array(self.actions), epochs=50, batch_size=512, verbose=False)
        self.states = []
        self.actions = []
        self.rewards = []

# class PolicyGradient:
    
#     def __init__(self, input_dim, output_dim, learning_rate=0.01, gamma=0.99, decay_rate=0.99):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.lr = learning_rate
#         self.gamma = gamma
#         self.decay_rate = decay_rate
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.build_model()
#         self.rms_prop_cache = {self.model.trainable_weights[i]: np.zeros(self.model.get_weights()[i].shape) for i in range(len(self.model.trainable_weights))}
        
        
#     def pg_loss(self, y_true, y_pred):
#         log_loss = K.categorical_crossentropy(y_true, y_pred)
#         return tf.reduce_mean(log_loss * self.discounted_rewards())
        
#     def build_model(self):
#         model = models.Sequential()
#         model.add(layers.Dense(64, activation='relu', input_shape=(self.input_dim,)))
#         model.add(layers.Dense(64, activation='relu'))
#         model.add(layers.Dense(self.output_dim, activation='softmax'))
#         #model.compile(optimizer='rmsprop', loss=self.pg_loss, metrics=['accuracy'])
#         model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#         self.model = model
        
#     def log_sar(self, s, a, r):
#         self.states.append(s)
#         actions = np.zeros(self.output_dim)
#         actions[a] = 1
#         self.actions.append(actions)
#         self.rewards.append(r)
        
#     def choose_action(self, state, legal_moves):
#         n_moves = len(legal_moves)
#         probs = self.model.predict(state[None, :])
#         return np.random.choice(n_moves, p=(probs.flatten()[:n_moves] / sum(probs.flatten()[:n_moves])))
#         #return np.random.choice(len(probs.flatten()), p=probs.flatten())
    
#     def discounted_rewards(self):
#         discounted_rewards = np.zeros(len(self.rewards))
#         total_rewards = 0
#         for t in reversed(range(len(self.rewards))):
#             total_rewards = total_rewards * self.gamma + self.rewards[t]
#             discounted_rewards[t] = total_rewards
#         discounted_rewards -= np.mean(discounted_rewards)
#         discounted_rewards /= np.std(discounted_rewards)
#         return discounted_rewards
    
#     def train(self):
#         split = int(np.array(self.states).shape[0]*0.2)
#         x_train = np.array(self.states)[split:]
#         y_train = np.array(self.actions)[split:]
#         x_val = np.array(self.states)[:split]
#         y_val = np.array(self.actions)[:split]
#         model.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val), callbacks=[early_stopping])
#         #self.states = []
#         #self.actions = []
#         #self.rewards = []
        
#     def get_gradients(self):
#         gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_weights)
#         input_tensors = [self.model.inputs[0], # input data
#                          self.model.sample_weights[0], # how much to weight each sample by
#                          self.model.targets[0], # labels
#                          K.learning_phase(), # train or test mode
#         ]
#         get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#         discounted_actions = (np.array(self.actions) * self.discounted_rewards()[:, None])
#         gradients_inputs = [np.array(self.states), [1], discounted_actions, 0]
#         return list(zip(self.model.trainable_weights, get_gradients(gradients_inputs)))
    
#     def update_weights(self):
#         grads = self.get_gradients()
#         for i, (key, val) in enumerate(grads):
#             self.rms_prop_cache[key] = self.decay_rate * self.rms_prop_cache[key] + (1 - self.decay_rate) * val**2
#             self.model.get_weights()[i] += self.lr * val / (np.sqrt(self.rms_prop_cache[key]) + 1e-5)
#         self.states = []
#         self.actions = []
#         self.rewards = []