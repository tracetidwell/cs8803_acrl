import numpy as np


def get_legal_moves(game, piece=None):
    if piece:
        return [list(move) for move in list(game.legalMoves(piece))]
    else:
        return [list(move) for move in list(game.legalMoves())]


def score_move(tetris, orient, slot, theta):
    field = np.frombuffer(tetris.getByteArray(), dtype=np.int32)
    new_field = np.array(field.reshape(21, 10))
    piece = tetris.getNextPiece()
    top = list(tetris.getTop())
    curr_height = top[slot]
    # print(piece, orient, slot)
    # print(tetris.getpWidth()[piece][orient])

    height = curr_height - tetris.getpBottom()[piece][orient][0]

    for i in range(1, tetris.getpWidth()[piece][orient]):
        
        # print(top[slot+i])
        # print(tetris.getpBottom()[piece][orient][i])
        # print('-----')
        height = max(height, top[slot+i] - tetris.getpBottom()[piece][orient][i])
    # print('-----')
    # print('-----')

    if height + tetris.getpHeight()[piece][orient] > tetris.getRows():
        lost = True
        return -100000

    for i in range(tetris.getpWidth()[piece][orient]):
        for j in range(height+tetris.getpBottom()[piece][orient][i], height+tetris.getpTop()[piece][orient][i]):
            new_field[j][i+slot] = tetris.getTurnNumber() + 1

    for c in range(tetris.getpWidth()[piece][orient]):
        top[slot+c] = height + tetris.getpTop()[piece][orient][c]

    rows_cleared = 0

    for r in range(height + tetris.getpHeight()[piece][orient] - 1, height - 1, -1):
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

    features = np.array([curr_height, rows_cleared, row_transitions, col_transitions, holes, wells, hole_depth, row_holes])
    return np.dot(features, theta)

gateway = JavaGateway()
tetris = gateway.entry_point.getState()
tetris.resetState()
theta = np.array([-12.63, 6.6, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04])
while not tetris.hasLost():
    scores = []
    moves = get_legal_moves(tetris)
    for orient, slot in moves:
        scores.append(score_move(tetris, orient, slot, theta))
    action = np.argmax(scores)
    reward = tetris.makeMove(int(action))