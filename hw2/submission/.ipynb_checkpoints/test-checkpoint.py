from py4j.java_gateway import JavaGateway
import numpy as np
import pickle
from utils import featurize_board, Tetris, play_game

def test_run(mu, tetris, feature_fn, iters=10):
    print('Test run')
    sum_reward = 0
    for itr in range(iters):
        print(f"Playing game #{itr}")
        rows_cleared = play_game(tetris, feature_fn, mu)
        sum_reward += rows_cleared

        print('Rows cleared: {}'.format(rows_cleared))
        print('------------------------------------')
    print('='*20)
    print(f'Average reward over {iters} runs is: {sum_reward/iters}')
    print('='*20)

if __name__ == "__main__":
    gateway = JavaGateway()
    tetris = gateway.entry_point.getState()
    tetris = Tetris(tetris)
    tetris.resetState()

    with open('weights/best_mu.pickle', 'rb') as mu_file:
        mu = pickle.load(mu_file)

    test_run(mu, tetris, featurize_board, iters=3)