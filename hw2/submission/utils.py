import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

    if not lost:

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
    return features

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