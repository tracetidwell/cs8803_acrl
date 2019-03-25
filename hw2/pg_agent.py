import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PGAgent:
    
    def __init__(self, input_dim, state_encoder, policy, value, gamma=0.9, scaler=StandardScaler):
        self.input_dim = input_dim
        self.state_encoder = state_encoder
        self.policy = policy
        self.value = value
        self.gamma = gamma
        self.scaler = scaler()
        self.n_episodes = 0
        self.states = []
        self.rewards = []
        self.gradients = []
        self.episode_states = []
        self.episode_rewards = []
        self.episode_gradients = []


    def encode_state(self, tetris, move):
        return self.state_encoder(tetris, move[0], move[1])


    def log_experience(self, state, reward):
        self.episode_states.append(state)
        self.episode_rewards.append(reward)


    def log_episode(self):
        self.states.append(self.episode_states)
        self.rewards.append(self.episode_rewards)
        self.gradients.append(self.episode_gradients)
        self.episode_states = []
        self.episode_rewards = []
        self.episode_gradients = []
        self.n_episodes += 1


    def choose_action(self, game):
        action, gradient = self.policy.choose_action(game)
        self.episode_gradients.append(gradient)
        return int(action)


    def choose_best_action(self, game):
        action, gradient = self.policy.choose_best_action(game)
        self.episode_gradients.append(gradient)
        return int(action)

    
    def update(self):
        avg_gradient = np.zeros(9)
        for i in range(self.n_episodes):
            discounted_rewards = np.array(self.rewards[i])
            for j in range(len(discounted_rewards)-2, -1, -1):
                discounted_rewards[j] += self.gamma * discounted_rewards[j+1]
            preds = self.value.predict(self.scaler.fit_transform(self.states[i]))
            delta = discounted_rewards - preds.reshape(-1,)
            avg_gradient += np.sum(delta.reshape(-1, 1) * np.array(self.gradients[i]), axis=0)
            self.value.train(self.scaler.fit_transform(self.states[i]), discounted_rewards)
        avg_gradient /= self.n_episodes
        self.policy.update_theta(avg_gradient * self.gamma)
        self.states = []
        self.rewards = []
        self.gradients = []
        self.n_episodes = 0



            # self.session.run(self.train_op, feed_dict={self.inputs: self.scaler.fit_transform(self.states[i]),
            #                                            self.labels: discounted_rewards})

            # for t in range(len(self.rewards[i])):
            #     g = 0
            #     for k in range(t+1, len(self.rewards[i])):
            #         g += self.gamma ** (k-t-1) * self.rewards[i][k]
            #     delta = g - self.value.predict(self.states[i][t])
            #     self.theta = self.theta + self.alpha * self.gamma ** t * delta * self.gradients[i][t]
        
