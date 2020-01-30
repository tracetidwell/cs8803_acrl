import numpy as np
from utils import featurize_board


class BoltzmannDistribution:

    
    def __init__(self, input_dim, output_dim, theta, state_encoder=featurize_board, alpha=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.theta = theta
        self.state_encoder = state_encoder
        self.alpha = alpha
        # self.gamma = gamma
        # self.n_episodes = 0
        # self.rewards = []
        # self.gradients = []
        # self.episode_rewards = []
        # self.episode_gradients = []


    def encode_state(self, tetris, move):
        return self.state_encoder(tetris, move[0], move[1])

            
    def choose_action(self, game):
        moves = game.get_legal_moves()
        #action_states = np.array([state_encoder(list(game.projectMove(orient, slot))) for orient, slot in moves]).T
        action_states = np.array([self.encode_state(game, move) for move in moves]).T
        probs = np.exp(np.matmul(self.theta, action_states))
        probs /= np.sum(probs)
        action = np.random.choice(len(moves), p=probs.flatten())
        gradient = action_states[:, action] - np.sum(action_states * probs, axis=1)
        #self.episode_gradients.append(gradient)
        return action, gradient


    def choose_best_action(self, game):
        moves = game.get_legal_moves()
        #action_states = np.array([state_encoder(list(game.projectMove(orient, slot))) for orient, slot in moves]).T
        action_states = np.array([self.encode_state(game, move) for move in moves]).T
        probs = np.exp(np.matmul(self.theta, action_states))
        probs /= np.sum(probs)
        action = np.argmax(probs.flatten())
        gradient = action_states[:, action] - np.sum(action_states * probs, axis=1)
        #self.episode_gradients.append(gradient)
        return action, gradient

    def update_theta(self, delta_theta):
        self.theta += (delta_theta * self.alpha)
    

    # def log_reward(self, reward):
    #     self.episode_rewards.append(reward)


    # def log_episode(self):
    #     self.rewards.append(self.episode_rewards)
    #     self.gradients.append(self.episode_gradients)
    #     self.episode_rewards = []
    #     self.episode_gradients = []
    #     self.n_episodes += 1

    
    # def update_theta(self):
    #     avg_gradient = np.zeros(9)
    #     for i in range(self.n_episodes):
    #     #     total_reward = sum(self.rewards[i])
    #     #     avg_gradient += np.sum(np.array(self.gradients[i]) * total_reward, axis=0)
    #     # avg_gradient /= self.n_episodes
    #     # self.theta += self.alpha * avg_gradient

    #         for t in range(len(self.rewards[i])):
    #             g = 0
    #             for k in range(t+1, len(self.rewards[i])):
    #                 g += self.gamma ** (k-t-1) * self.rewards[i][k]
    #             delta = g - value
    #             self.theta = self.theta + self.alpha * self.gamma ** t * delta * self.gradients[i][t]
        
    #     self.rewards = []
    #     self.gradients = []
    #     self.n_episodes = 0

