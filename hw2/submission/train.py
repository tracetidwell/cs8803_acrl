import os
from py4j.java_gateway import JavaGateway
import numpy as np
import pickle
from utils import featurize_board, Tetris, play_game

def play_game(tetris, feature_fn, theta, verbose=False):
    tetris.resetState()
    while not tetris.hasLost():
    
        scores = []
        moves = tetris.get_legal_moves()
        for orient, slot in moves:
            features = feature_fn(tetris, orient, slot)
            score = np.dot(features,theta)
            scores.append(score)

        action = int(np.argmax(scores))
        reward = tetris.makeMove(action)

        if verbose:
            if tetris.getTurnNumber() % verbose == 0:
                print('Lines cleared after {} turns: {}'.format(tetris.getTurnNumber(), tetris.getRowsCleared()))
    if verbose:
        print('Total lines cleared after {} turns: {}'.format(tetris.getTurnNumber(), tetris.getRowsCleared()))
        print('--------------------------------------------------------------------------------')
        
    return tetris.getRowsCleared()

def initialize_mu_sigma(n=15, size=9, init_mu=0, init_sigma=0.1):
    if 'mu.pickle' not in os.listdir():
        thetas = []
        rewards = []
        while len(thetas) < n:
            theta = np.random.normal(init_mu, init_sigma, size)
            reward = play_game(tetris, featurize_board, theta)
            if reward > 0:
                thetas.append(theta)
                rewards.append(reward)
                
        return np.mean(thetas, axis=0), np.std(thetas, axis=0)
    
    else:
        with open('mu.pickle', 'rb') as mu_file:
            mu = pickle.load(mu_file)
        with open('sigma.pickle', 'rb') as sigma_file:
            sigma = pickle.load(sigma_file)
            
        return mu, sigma

def save_mean_rewards(mean_rewards):
    if 'mean_rewards.pickle' not in os.listdir():
        with open('mean_rewards.pickle', 'wb') as rewards_file:
            pickle.dump(mean_rewards, rewards_file)
    else:
        with open('mean_rewards.pickle', 'rb') as rewards_file:
            old_mean_rewards = pickle.load(rewards_file)
        new_mean_rewards = old_mean_rewards + mean_rewards
        with open('mean_rewards.pickle', 'wb') as rewards_file:
            pickle.dump(new_mean_rewards, rewards_file)

def save_params(mu, sigma):
    with open('mu.pickle', 'wb') as mu_file:
        pickle.dump(mu, mu_file)
    with open('sigma.pickle', 'wb') as sigma_file:
        pickle.dump(sigma, sigma_file)

def noisy_cross_entropy(mu, sigma, n_samples, n_features, rho, n_games, noise, score_fn, tetris, feature_fn, 
                        max_iter=1000, epsilon=0.01):
    
    converged = False
    n_iter = 0
    mean_rewards = []
    thetas = np.random.normal(mu, sigma, (n_samples, n_features))
    
    while not converged and n_iter < max_iter:
        rewards = []
        
        print('Iteration: {}'.format(n_iter+1))
        for i in range(n_samples):
            if i % 10 == 0:
                print('Running sample {}'.format(i))
            rewards.append(score_fn(tetris, feature_fn, thetas[i]))
            
        children = thetas[np.argsort(rewards)][-rho:]
        new_mu = np.mean(children, axis=0)
        
        converged = np.all(abs(new_mu - mu) < epsilon)
            
        mu = new_mu
        sigma = np.std(children, axis=0) + noise
        thetas = np.random.normal(mu, sigma, (n_samples, n_features))
        
        print('Calculating mean')
        reward = 0
        for i in range(n_games):
            if i % 5 == 0:
                print('Running mean sample {}'.format(i))
            reward += score_fn(tetris, feature_fn, mu)
            
        mean_rewards.append(reward / n_games)
        
        n_iter += 1
        
        print('Average rows cleared after {} iterations: {}'.format(n_iter, mean_rewards[-1]))
        print('------------------------------------')
        print('------------------------------------')
        
        save_params(mu, sigma)
        save_mean_rewards(mean_rewards)
        
    return mu, sigma, mean_rewards

if __name__ == "__main__":
    gateway = JavaGateway()
    tetris = gateway.entry_point.getState()
    tetris = Tetris(tetris)
    tetris.resetState()
    
    n_samples = 50
    n_features = 9
    rho = 10
    n_games = 15
    noise = 0.2
    mu, sigma = initialize_mu_sigma(rho, n_features, 0, 0.1)

    new_mu, new_sigma, mean_lines = noisy_cross_entropy(mu, sigma, n_samples, n_features, rho, n_games, 
                                                    noise, play_game, tetris, featurize_board)