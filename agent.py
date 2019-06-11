import numpy as np
import pandas as pd
import pickle

import chess
import chess.engine
import chess.variant

import functools
from collections import Counter, OrderedDict, defaultdict

from tensorflow.keras.models import load_model

from analysis import feature_extractor2 #, feature_weights

infinity = float('inf')

fit_estim = None
tf_model = None


def r_squared(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


from tensorflow.python.keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"r_squared": r_squared})


def utility1(board, print_board=False):
    global node_count
    node_count += 1

    factor = (1 if (board.turn == chess.WHITE) else -1)

    if board.is_variant_win():
        return 1000 * factor
    elif board.is_variant_loss():
        return -1000 * factor

    fen = Counter(board.board_fen())
    wk_rank = chess.square_rank(board.pieces(chess.KING, chess.WHITE).pop())
    bk_rank = chess.square_rank(board.pieces(chess.KING, chess.BLACK).pop())
    rankdiff = (wk_rank - bk_rank)
    material = (9 * (fen['Q'] - fen['q']) + 5 * (fen['R'] - fen['r']) +
                3 * (fen['B'] - fen['b'] + fen['N'] - fen['n']))

    w_mobility = len(list(board.legal_moves))
    board.push(chess.Move.null())
    b_mobility = len(list(board.legal_moves))
    board.pop()
    mobility = (w_mobility - b_mobility) * (-1 if board.turn == chess.BLACK else 1)
    #mobility = 0

    # utility from white's perspective
    utility = material + 5 * rankdiff + 0.2 * mobility

    return utility * factor


def utility2(board):
    global node_count, fit_estim
    node_count += 1

    if not fit_estim:
        with open('./notebooks/estimator.pkl', 'rb') as f:
            print("loading estimator")
            fit_estim = pickle.load(f)

    factor = (1 if (board.turn == chess.WHITE) else -1)

    if board.is_variant_win():
        return 1000 * factor
    elif board.is_variant_loss():
        return -1000 * factor

    fen = board.board_fen()
    features = feature_extractor2(fen, board)
    features = pd.DataFrame(features, index=[0])
    score = fit_estim.predict(features)[0]
    if node_count % 50 == 0:
        print(f"Nodes: {node_count}")

    return factor * score


def utility3(board):
    global node_count, tf_model
    node_count += 1

    if not tf_model:
        print("loading tf model")
        tf_model = load_model('./notebooks/tf_model3.ckpt')

    factor = (1 if (board.turn == chess.WHITE) else -1)

    if board.is_variant_win():
        return 1000 * factor
    elif board.is_variant_loss():
        return -1000 * factor

    fen = board.board_fen()
    features = feature_extractor2(fen, board)
    features = pd.DataFrame(features, index=[0])
    score = tf_model.predict(features)[0]
    if node_count % 50 == 0:
        print(f"Nodes: {node_count}")

    return factor * score


class Game:
    def __init__(self, engine):
        self.engine = engine

    def terminal_state(self, board):
        return board.is_game_over()

    def oracle(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=0.100))
        return result.move

    def alphabeta(self, board):
        global node_count
        node_count = 0
        result = alphabeta_search(
            self, board, 2,
            lambda state: utility1(state))
        node_counts.append(node_count)
        return result

    def fit_alphabeta(self, board):
        global node_count
        node_count = 0
        result = alphabeta_search(
            self, board, 2,
            lambda state: utility3(state))
        node_counts.append(node_count)
        # board.push(result)
        # info = self.engine.analyse(board, chess.engine.Limit(time=0.100))
        # print("Score: {}".format(info['score']))
        # board.pop()
        return result

    def successors(self, board):
        states = []
        for m in board.legal_moves:
            b = board.copy()
            b.push(m)
            states.append((m, b))
        return states


def fixed_beam(states, eval_fn, width, reverse=False):
    factor = 1.0 if reverse else -1.0

    s_len = len(states)

    if s_len <= 1:
        return states
    else:
        rng = np.arange(s_len)
        width = min(width, s_len)
        scores = [eval_fn(states[i][1]) * factor for i in rng]
        beam = np.flip(np.argsort(scores), axis=0)[:width]
        return [states[i] for i in beam]

def stochastic_beam(states, eval_fn, width, reverse=False, batch=None):
    factor = 1.0 if reverse else -1.0

    s_len = len(states)

    batch = batch or width

    if s_len <= width:
        return fixed_beam(states, eval_fn, s_len, reverse=reverse)

    rng = np.arange(s_len)

    width = min(width, s_len)
    scores = np.tile(-float('inf'), s_len)
    beam = np.random.choice(rng, size=width, replace=False)

    for i in beam:
        scores[i] = eval_fn(states[i][1]) * factor

    while True:
        probs = ~np.isfinite(scores)
        sprob = sum(probs)
        probs = probs / sprob
        size = min(batch, (probs > 0).sum())
        if size < 1:
            break
        choices = np.random.choice(rng, size=size, replace=False, p=probs)
        for i in choices:
            scores[i] = eval_fn(states[i][1]) * factor
        old = beam
        beam = np.flip(np.argsort(scores), axis=0)[:width]
        if (beam == old).all() or np.isfinite(scores).all():
            break

    return [states[i] for i in beam]

node_count = 0
node_counts = []

def alphabeta_search(game, state, max_depth, eval_fn, beam=None):
    """
    Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function.
    """
    def cutoff_test(state, depth):
        return depth <= 0 or game.terminal_state(state)

    def do_eval(state):
        return eval_fn(state)

    def max_value(state, lower_bound, upper_bound, depth):
        if cutoff_test(state, depth):
            return do_eval(state), None
        max_action = None
        max_score = -infinity
        succ = game.successors(state)

        if beam:
            succ1 = stochastic_beam(succ, eval_fn, beam, reverse=False)
            # test = fixed_beam(succ, eval_fn, beam, reverse=False)
            # succ = succ1
            # s1 = {m[0] for m in test}
            # s2 = {m[0] for m in succ}
            # print(f"Overlap: {len(s1.intersection(s2))} / {beam}")

        for (action, s_next) in succ:
            v, _ = min_value(s_next, lower_bound, upper_bound, depth - 1)
            if v > max_score:
                max_score = v
                max_action = action
            lower_bound = max(lower_bound, v)  # raise/tighten lower bound
            if lower_bound >= upper_bound:
                break
        # print(f"Max at depth {depth} = {v} / alpha = {lower_bound}")
        return max_score, max_action

    def min_value(state, lower_bound, upper_bound, depth):
        if cutoff_test(state, depth):
            return do_eval(state), None
        min_action = None
        min_score = infinity
        succ = game.successors(state)

        if beam:
            succ1 = stochastic_beam(succ, eval_fn, beam, reverse=True)
            # test = fixed_beam(succ, eval_fn, beam, reverse=True)
            # succ = succ1
            # s1 = {m[0] for m in test}
            # s2 = {m[0] for m in succ}
            # print(f"Overlap: {len(s1.intersection(s2))} / {beam}")

        for (action, s_next) in succ:
            v, _ = max_value(s_next, lower_bound, upper_bound, depth - 1)
            if v < min_score:
                min_score = v
                min_action = action
            upper_bound = min(upper_bound, v)  # lower/tighten upper bound
            if upper_bound <= lower_bound:
                break
        # print(f"Min at depth {depth} = {v} / beta = {upper_bound}")
        return min_score, min_action

    _, a = max_value(state, -infinity, infinity, max_depth)
    print("AB Score={}".format(_))
    return a


def make_move(game, board, strategy):
    return getattr(game, strategy)(board)


def time_utility1(board):
    import time
    t0 = time.time()
    for i in range(1000):
        utility1(board)
    elapsed = time.time() - t0
    print(f"Rate states/second = {1000/elapsed}")


def time_utility2(board):
    import time
    t0 = time.time()
    global fit_estim
    if not fit_estim:
        with open('./notebooks/estimator.pkl', 'rb') as f:
            print("loading estimator")
            fit_estim = pickle.load(f)
    for i in range(1000):
        utility2(board)
    elapsed = time.time() - t0
    print(f"Rate states/second = {1000/elapsed}")


def main(w_strat='fit_alphabeta', b_strat='fit_alphabeta'):
    engine = chess.engine.SimpleEngine.popen_uci('../Stockfish/src/stockfish')

    try:
        print(f"Opened engine {engine.id.get('name')}")

        game = Game(engine)
        board = chess.variant.RacingKingsBoard()

        print(f"Playing {w_strat} vs {b_strat}")

        while True:
            strategy = w_strat if board.turn else b_strat
            m = make_move(game, board, strategy)
            t = "WHITE" if board.turn else "BLACK"
            print(f"Move: {t} {m}")
            # print("States eval {}".format(node_counts[-1]))
            board.push(m)
            print(board)
            if board.is_variant_draw() or board.is_game_over():
                break

        if board.is_variant_draw():
            result = '1/2-1/2'
        else:
            result = board.result()

        print(f"Game over: {w_strat} {result} {b_strat}")

    finally:
        engine.quit()


if __name__ == '__main__':
    main()
    # time_utility2(chess.variant.RacingKingsBoard())
