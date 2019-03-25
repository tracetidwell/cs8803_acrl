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


class FeatureFunction:

    def __init__(self, theta):
        self.theta = theta

    def log_experience(self, s, a, r, next_s, d):
        pass

    def encode_state(self, tetris):
        pass

    def score_move(tetris, orient, slot, theta):
        lost = False
        rows_cleared = 0
        field = np.frombuffer(tetris.getByteArray(), dtype=np.int32)
        new_field = np.array(field.reshape(21, 10))
        piece = tetris.getNextPiece()
        top = list(tetris.getTop())
        curr_height = top[slot]

        height = curr_height - tetris.getpBottom()[piece][orient][0]

        for i in range(1, tetris.getpWidth()[piece][orient]):
            height = max(height, top[slot+i] - tetris.getpBottom()[piece][orient][i])

        if height + tetris.getpHeight()[piece][orient] >= tetris.getRows():
            lost = True

        for i in range(tetris.getpWidth()[piece][orient]):
            for j in range(height+tetris.getpBottom()[piece][orient][i], min(21, height+tetris.getpTop()[piece][orient][i])):
                new_field[j][i+slot] = tetris.getTurnNumber() + 1

        for c in range(tetris.getpWidth()[piece][orient]):
            top[slot+c] = height + tetris.getpTop()[piece][orient][c]

        if not lost:
            for r in range(min(20, height + tetris.getpHeight()[piece][orient] - 1), height - 1, -1):
                full = True
                for c in range(tetris.getCols()):
                    if new_field[r][c] == 0:
                        full = False
                        break
                        
                if full:
                    rows_cleared += 1
                    for c in range(tetris.getCols()):
                        for i in range(r, top[c]):
                            new_field[i][c] = new_field[i+1][c]
                        top[c] -= 1
                        while top[c] >= 1 and new_field[top[c]-1][c] == 0:
                            top[c] -= 1

        row_transitions = 0
        for r in range(21):
            for c in range(10):
                if new_field[r, c] == 0:
                    if new_field[r, max(0, c-1)] != 0:
                        row_transitions += 1
                    if new_field[r, min(9, c+1)] != 0:
                        row_transitions += 1

        col_transitions = 0
        for c in range(10):
            for r in range(21):
                if new_field[r, c] == 0:
                    if new_field[max(0, r-1), c] != 0:
                        col_transitions += 1
                    if new_field[min(20, r+1), c] != 0:
                        col_transitions += 1

        holes = 0
        row_holes = 0
        hole_depth = 0
        for r in range(20):
            row_hole = False
            for c in range(10):
                if new_field[r, c] == 0:
                    if new_field[r+1, c] != 0:
                        holes += 1
                        hole_depth += 1
                        i = 2
                        while r+i < 20 and new_field[r+i, c] != 0:
                            hole_depth += 1
                            i += 1
                        row_hole = True
            if row_hole:
                row_holes += 1

        wells = 0
        for c in range(10):
            if c == 0:
                if top[c+1] > top[c]:
                    wells += sum(range(top[c+1] - top[c] + 1))
            elif c == 9:
                if top[c-1] > top[c]:
                    wells += sum(range(top[c-1] - top[c] + 1))
            else:
                if top[c+1] > top[c] and top[c-1] > top[c]:
                    min_diff = min(top[c+1] - top[c], top[c-1] - top[c])
                    wells += sum(range(min_diff + 1))

        if lost:
            features = np.array([curr_height, rows_cleared, row_transitions, col_transitions, holes, wells, hole_depth, row_holes, 1])
        else:
            features = np.array([curr_height, rows_cleared, row_transitions, col_transitions, holes, wells, hole_depth, row_holes, 0])

        return np.dot(features, theta)

    def choose_action(self, tetris, state):
        scores = []
        for orient, slot in tetris.get_legal_moves():
            score = score_move(tetris, orient, slot, self.theta)
            scores.append(score)
        return np.argmax(scores)


class DeepQNetwork:

    def __init__(self, session, input_dim, output_dim, buffer_size, state_encoder, min_experience=64, 
                 gamma=0.99, gamma_decay=0.99, epsilon=0.99, epsilon_decay=0.9):
        self.session = session
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.buffer_size = buffer_size
        self.state_encoder = state_encoder
        self.min_experience = min_experience
        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # def dqn_loss(actions):

        #     def loss(self, y_true, y_pred):
        #         action_vals = tf.reduce_sum(y_pred * tf.one_hot(actions, self.output_dim))
        #         return tf.reduce_sum(tf.square(y_true - action_prob))

        #     return loss

        
        self.targets = tf.placeholder(tf.float32, shape=(None,), name='targets')
        self.actions_ = tf.placeholder(tf.int32, shape=(None,), name='actions')

        #self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        self.inputs = models.Input(shape=(self.input_dim,))
        self.fc1 = layers.Dense(512, activation='relu')(self.inputs)
        self.fc2 = layers.Dense(256, activation='relu')(self.fc1)
        self.fc3 = layers.Dense(128, activation='relu')(self.fc2)
        self.fc4 = layers.Dense(64, activation='relu')(self.fc3)
        self.predictions = layers.Dense(self.output_dim, activation=None)(self.fc4)
        self.model = models.Model(inputs=self.inputs, outputs=self.predictions)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        self.init_loss = loss = tf.reduce_sum(tf.square(self.targets - self.predictions))
        self.init_op = tf.train.RMSPropOptimizer(1e-2).minimize(self.init_loss)
        
        self.action_vals = tf.reduce_sum(self.predictions * tf.one_hot(self.actions_, self.output_dim))
        self.train_loss = tf.reduce_sum(tf.square(self.targets - self.action_vals))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.train_loss)

        # model = models.Sequential()
        # model.add(layers.Dense(512, activation='relu', input_shape=(self.input_dim,)))
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(34))
        # #model.compile(optimizer='adam', loss=dqn_loss(actions), metrics=['mae'])
        # model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # self.model = model

        # self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='inputs')
        # self.targets = tf.placeholder(tf.float32, shape=(None,), name='targets')
        # self.actions_ = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # action_vals = tf.reduce_sum(self.model.outputs[0] * tf.one_hot(self.actions_, self.output_dim))
        # loss = tf.reduce_sum(tf.square(self.targets - action_vals))
        # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

    # def initialize(self, x_train, y_train, epochs, batch_size, validation_data, callbacks):
    #     self.model.fit(x_train, y_train, epochs=epochs, validation_data=validation_data, batch_size=batch_size, callbacks=callbacks)

    def initialize(self, states, target_Qs, actions):
        self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                   self.targets: target_Qs,
                                                   self.actions_: actions})

    def predict(self, x):
        return self.model.predict(x)

    def train(self, batch_size=256):
        if len(self.states) < self.min_experience:
            return
        batch_size = min(batch_size, len(self.states))
        indexes = np.random.choice(len(self.states), batch_size, replace=False)
        states = np.array(self.states).reshape(-1, self.input_dim)[indexes]
        actions = np.array(self.actions)[indexes]
        rewards = np.array(self.rewards)[indexes]
        next_states = np.array(self.next_states).reshape(-1, self.input_dim)[indexes]
        dones = np.array(self.dones)[indexes]
        Qs_next_state = self.predict(next_states)
        target_Qs = []
        
        for i in range(batch_size):
            if dones[i]:
                target_Qs.append(rewards[i])
            else:
                target_Qs.append(rewards[i] + self.gamma * np.max(Qs_next_state[i]))
                
        self.session.run(self.train_op, feed_dict={self.inputs: states,
                                                   self.targets: target_Qs,
                                                   self.actions_: actions})

    def choose_action(self, tetris, state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return np.random.choice(len(tetris.get_legal_moves())-1)
        else:
            return np.argmax(self.predict(state)[0])

    def reset_epsilon(self, epsilon):
        self.epsilon = epsilon

    def log_experience(self, s, a, r, next_s, d):
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

    def encode_state(self, tetris):
        return self.state_encoder.transform(tetris)


class DQN:
    
    def __init__(self, input_dim, output_dim, buffer_size, sess, state_encoder, 
                 alpha=0.001, gamma=0.99, gamma_decay=0.99, epsilon=1, epsilon_decay=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.buffer_size = buffer_size
        self.session = sess
        self.piece_encoder = piece_encoder
        self.board_encoder = board_encoder
        self.alpha = alpha
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
        self.labels = tf.placeholder(tf.float32, shape=(None,self.output_dim), name='labels')
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
        
        self.fc1 = layers.Dense(512, activation='relu')(self.inputs)
        self.fc2 = layers.Dense(256, activation='relu')(self.fc1)
        self.fc3 = layers.Dense(128, activation='relu')(self.fc2)
        self.fc4 = layers.Dense(64, activation='relu')(self.fc3)
        self.output = layers.Dense(self.output_dim, activation=None)(self.fc4)

        init_loss = loss = tf.reduce_sum(tf.square(self.labels - self.output))
        self.init_op = tf.train.RMSPropOptimizer(1e-2).minimize(init_loss)
        
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
        
        
    def predict(self, states):
        return self.session.run(self.output, feed_dict={self.inputs: states})
        #return self.model.predict(X)


    def initialize(self, states, scores, epochs, batch_size):
        n_batches = states.shape[0] // batch_size
        for i in range(epochs):
            start = 0
            for i in range(n_batches-1):
                states_batch = states[start:start+batch_size]
                scores_batch = scores[start:start+batch_size]
                start += batch_size
                self.session.run(self.init_op, feed_dict={self.inputs: states_batch,
                                                          self.labels: scores_batch})
            states_batch = states[start:]
            scores_batch = scores[start:]
            self.session.run(self.init_op, feed_dict={self.inputs: states_batch,
                                                        self.labels: scores_batch})
            

    
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
                target_Qs.append(rewards[i] + self.gamma * np.max(Qs_next_state[i]))
                
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



class Agent:
    
    def __init__(self, input_dim, output_dim, state_encoder, policy=Boltzmann_distribution, theta=[], alpha=0.001):
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
        self.policy = policy(input_dim, output_dim, theta)
        self.state_encoder = state_encoder
        #self.board_encoder = OneHotEncoder(categories='auto')
        #self.board_encoder.fit(np.array(range(state_values)).reshape(-1, 1))
        #self.piece_encoder = OneHotEncoder(categories='auto')
        #self.piece_encoder.fit(np.array(range(game.getPieces())).reshape(-1, 1))
        #self.piece_encoder = {0: [0, 0, 0],
                            #   1: [0, 0, 1],
                            #   2: [0, 1, 0],
                            #   3: [0, 1, 1],
                            #   4: [1, 0, 0], 
                            #   5: [1, 0, 1], 
                            #   6: [1, 1, 0]}
    
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