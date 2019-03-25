import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


def batch_generator(x, y, batch_size):
    n_batches = len(x) // batch_size
    batch = 0
    
    while True:
    
        if batch < n_batches - 1:
            yield x[batch*batch_size:batch*batch_size+batch_size], y[batch*batch_size:batch*batch_size+batch_size]
            batch += 1
        else:
            yield x[batch*batch_size:], y[batch*batch_size:]
            batch = 0


def initialize_session_vars(sess):
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init_g)
    sess.run(init_l)


def bertsekas_features(tetris, orient, slot):
    ROWS = tetris.getRows()
    COLS = tetris.getCols()
    piece = tetris.getNextPiece()
    WIDTH = tetris.getpWidth()[piece][orient]
    lost = False
    field = np.frombuffer(tetris.getByteArray(), dtype=np.int32)
    new_field = np.array(field.reshape(21, 10))
    piece = tetris.getNextPiece()
    top = list(tetris.getTop())
    
    height = np.max(np.array(top[slot:slot+WIDTH]) - np.array(list(tetris.getpBottom()[piece][orient])[:WIDTH]))

    if height + tetris.getpHeight()[piece][orient] >= tetris.getRows():
        lost = True

    for i in range(WIDTH):
        for j in range(height+tetris.getpBottom()[piece][orient][i], min(21, height+tetris.getpTop()[piece][orient][i])):
            new_field[j][i+slot] = tetris.getTurnNumber() + 1

    top[slot:slot+WIDTH] = list(tetris.getpTop()[piece][orient][:WIDTH]) + height    

    if not lost:

        for r in range(min(ROWS-1, height + tetris.getpHeight()[piece][orient] - 1), height - 1, -1):
            full = True
            for c in range(COLS):
                if new_field[r][c] == 0:
                    full = False
                    break

            if full:
                for c in range(COLS):
                    for i in range(r, top[c]):
                        new_field[i][c] = new_field[i+1][c]
                    top[c] -= 1
                    while top[c] >= 1 and new_field[top[c]-1][c] == 0:
                        top[c] -= 1

    holes = 0
    for i in range(min(ROWS-1, max(top))):
        for j in range(10):
            if new_field[i, j] == 0 and new_field[i+1, j] != 0:
                holes += 1

    features = top + list(abs(np.array(top)[:-1] - np.array(top)[1:])) + [max(top)] + [holes]
    return np.array(features)


def featurize_board(tetris, orient, slot):
    ROWS = tetris.getRows()
    COLS = tetris.getCols()
    piece = tetris.getNextPiece()
    WIDTH = tetris.getpWidth()[piece][orient]
    lost = False
    rows_cleared = 0
    eroded_piece_cells = 0
    field = np.frombuffer(tetris.getByteArray(), dtype=np.int32)
    new_field = np.array(field.reshape(21, 10))
    piece = tetris.getNextPiece()
    top = list(tetris.getTop())
    
    height = np.max(np.array(top[slot:slot+WIDTH]) - np.array(list(tetris.getpBottom()[piece][orient])[:WIDTH]))

    if height + tetris.getpHeight()[piece][orient] >= tetris.getRows():
        lost = True

    for i in range(WIDTH):
        for j in range(height+tetris.getpBottom()[piece][orient][i], min(21, height+tetris.getpTop()[piece][orient][i])):
            new_field[j][i+slot] = tetris.getTurnNumber() + 1

    top[slot:slot+WIDTH] = list(tetris.getpTop()[piece][orient][:WIDTH]) + height

    #landing_height = np.min(np.where(new_field==tetris.getTurnNumber()+1)[0]) + (tetris.getpHeight()[piece][orient] / 2)
    

    if not lost:
        # mask = np.all(new_field, axis=1)
        # rows_cleared = sum(mask)
        # new_field = np.concatenate([new_field[~mask], np.zeros((rows_cleared, 10))])
        # top = 21 - np.argmax(new_field[::-1, :], axis=0)
        # eroded_piece_cells = rows_cleared * (4 - sum(new_field.flatten() == tetris.getTurnNumber()+1))
        

        for r in range(min(ROWS-1, height + tetris.getpHeight()[piece][orient] - 1), height - 1, -1):
            full = True
            for c in range(COLS):
                if new_field[r][c] == 0:
                    full = False
                    break

            if full:
                rows_cleared += 1
                for c in range(COLS):
                    for i in range(r, top[c]):
                        new_field[i][c] = new_field[i+1][c]
                    top[c] -= 1
                    while top[c] >= 1 and new_field[top[c]-1][c] == 0:
                        top[c] -= 1

        eroded_piece_cells = rows_cleared * (4 - sum(new_field.flatten() == tetris.getTurnNumber()+1))

    landing_height = np.where(new_field==tetris.getTurnNumber()+1)[0].mean()
    max_height = max(top)

    holes = 0
    row_holes = 0
    hole_depth = 0
    row_transitions = 0
    for i in range(min(ROWS-1, max_height)):
        row_hole = False
        for j in range(10):
            if new_field[i, j] == 0:
                if j < 9 and new_field[i, j+1] != 0:
                    row_transitions += 1
                if new_field[i+1, j] != 0:
                    holes += 1
                    row_hole = True
                    hole_depth += 1
                    counter = 2
                    while i + counter < min(max_height, 20) and new_field[i+counter, j] != 0:
                        hole_depth += 1
                        counter += 1
            elif j < 9 and new_field[i, j] != 0 and new_field[i, j+1] == 0:
                row_transitions += 1
        if row_hole:
            row_holes += 1
    row_transitions += sum(new_field[:max_height, 0] == 0)
    row_transitions +=sum(new_field[:max_height, -1] == 0)

    wells = 0
    col_transitions = 0
    for c in range(COLS):
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
        for r in range(min(ROWS, max(top)+1)):
            if new_field[r, c] == 0:
                if r == 0:
                    col_transitions += 1
                if new_field[max(0, r-1), c] != 0:
                    col_transitions += 1
                if new_field[min(20, r+1), c] != 0:
                    col_transitions += 1

    features = np.array([landing_height, eroded_piece_cells, row_transitions, col_transitions, holes, wells, hole_depth, row_holes, 2*lost])
    #features = np.array([landing_height, eroded_piece_cells, row_transitions, col_transitions, holes, wells, hole_depth, row_holes])

    return features


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

    holes = 0
    row_holes = 0
    hole_depth = 0
    row_transitions = 0
    for r in range(20):
        row_hole = False
        for c in range(10):
            if new_field[r, c] == 0:
                if new_field[r, max(0, c-1)] != 0:
                    row_transitions += 1
                if new_field[r, min(9, c+1)] != 0:
                    row_transitions += 1
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
    col_transitions = 0
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
        for r in range(21):
            if new_field[r, c] == 0:
                if new_field[max(0, r-1), c] != 0:
                    col_transitions += 1
                if new_field[min(20, r+1), c] != 0:
                    col_transitions += 1

    features = np.array([curr_height, rows_cleared, row_transitions, col_transitions, holes, wells, hole_depth, row_holes, 1*lost])

    return rows_cleared * 0.1, np.dot(features, theta)


def generate_training_data(n_games, tetris, theta, state_encoder):
    examples = []
    targets = []
    for i in range(n_games):
        tetris.resetState()
        episode_rewards = []
        while not tetris.hasLost():
            scores = []
            rewards = []
            moves = get_legal_moves(tetris)
            for orient, slot in moves:
                reward, score = score_move(tetris, orient, slot, theta)
                rewards.append(reward)
                scores.append(score)
            examples.append(state_encoder.transform(tetris))
            episode_rewards.append(np.concatenate([rewards, np.zeros(34-len(rewards))]))
            action = int(np.argmax(scores))
            reward = tetris.makeMove(action)
        discounted_rewards = np.array(episode_rewards)
        for i in range(len(discounted_rewards)-2, -1, -1):
            discounted_rewards[i, :] += 0.9 * max(discounted_rewards[i+1, :])
        targets += list(discounted_rewards)
    return pd.DataFrame(np.concatenate([np.array(examples).reshape(-1, 217), np.array(targets)], axis=1))


def update_training_data(new_df):
    old_df = pd.read_csv('training_data.csv')
    updated_df = pd.DataFrame(np.concatenate([old_df.values, new_df.values]))
    updated_df.to_csv('training_data.csv', index=False)


class PieceEncoder:

    def __init__(self, n):
        self.n = n
        self.encoder = OneHotEncoder(categories='auto', sparse=False)
        self.encoder.fit(np.array(range(n)).reshape(-1, 1))

    def transform(self, x):
        return self.encoder.transform(np.array([x]).reshape(1, -1)).reshape(-1,)


class BoardEncoder:

    def transform(self, tetris):
        field = np.frombuffer(tetris.getByteArray(), dtype=np.int32)
        return np.where(field>0, 1, 0)


class StateEncoder:

    def __init__(self, n):
        self.piece_encoder = PieceEncoder(n)
        self.board_encoder = BoardEncoder()

    def transform(self, tetris):
        return np.concatenate([self.piece_encoder.transform([tetris.getNextPiece()]), self.board_encoder.transform(tetris)]).reshape(1, -1)


class Tetris:

    def __init__(self, tetris):
        self.tetris = tetris

    def __getattr__(self, item):
        return getattr(self.tetris, item)

    def get_legal_moves(self, piece=None):
        if piece:
            return [list(move) for move in list(self.legalMoves(piece))]
        else:
            return [list(move) for move in list(self.legalMoves())]

    def get_board(self):
        field = np.frombuffer(self.getByteArray(), dtype=np.int32)
        return field.reshape(self.getRows(), self.getCols())[::-1, :]


def play_game(tetris, agent, visualize=False):
    tetris.resetState()
    total_reward = 0
    while not tetris.hasLost():
        if visualize:
            field = np.frombuffer(tetris.getByteArray(), dtype=np.int32)
            print(field.reshape(21, 10)[::-1, :])
            print('  ')
        state = agent.encode_state(tetris)
        action = int(agent.choose_action(tetris, state))
        reward = tetris.makeMove(action)
        next_state = agent.encode_state(tetris)
        done = tetris.hasLost()

        total_reward += reward

        agent.log_experience(state, action, reward, next_state, done)
        agent.train()

    return total_reward
