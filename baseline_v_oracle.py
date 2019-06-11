import chess
import chess.engine
import chess.variant
import random
import time

import functools
from heapq import heappush, heappop
from enum import Enum
from collections import Counter


max_nodes = 1000
infinity = 1.0e400


class Game:
    def __init__(self, engine, opponent=chess.BLACK):
        self.engine = engine
        self.opponent = opponent

    def to_move(self, board):
        return board.turn

    def terminal_state(self, board):
        return board.is_game_over()

    def oracle(self, board):
        result = self.engine.play(board, chess.engine.Limit(depth=1)) # nodes = 100
        return result.move

    def successors(self, board):
        states = []
        for m in board.legal_moves:
            b = board.copy()
            b.push(m)
            states.append((m, b))
        return states

    functools.lru_cache(maxsize=1024)
    def utility(self, board, player):
        #info = self.engine.analyse(board, chess.engine.Limit(time=0.001)) # nodes = 100
        #score = info['score'].pov(player).score()
        t = time.time()
        fen = Counter(board.board_fen())
        wk_rank = chess.square_rank(board.pieces(chess.KING, chess.WHITE).pop())
        bk_rank = chess.square_rank(board.pieces(chess.KING, chess.BLACK).pop())
        rankdiff = (wk_rank - bk_rank)
        material = 9 * (fen['Q'] - fen['q']) + 5 * (fen['R'] - fen['r']) + 3 * (fen['B'] - fen['b'] + fen['N'] - fen['n']) + (fen['P'] - fen['p'])

        w_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        b_mobility = len(list(board.legal_moves))
        board.pop()
        mobility = (w_mobility - b_mobility) * (-1 if board.turn == chess.BLACK else 1)

        # utility from white's perspective
        utility = material + 5 * rankdiff + 0.2 * mobility

        utility *= (1 if (player == chess.WHITE) else -1)

        #print(board)
        #print(f"WHITE? {player == chess.WHITE} Util: {utility} Depth: {len(board.move_stack)} Elapsed: {time.time() - t}")
        #import pdb; pdb.set_trace()

        return utility


# source, modified from: http://aima.cs.berkeley.edu/python/games.html
def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best = seq[0]
    best_score = fn(best);
    n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                best = x
    return best

def argmax(seq, fn):
    return argmin(seq, lambda x: -fn(x))

def alphabeta_search(state, game, max_depth=2):
    """
    Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function.
    """

    player = game.to_move(state)

    def eval_fn(state):
        return game.utility(state, player)

    def cutoff_test(state, depth):
        return depth > max_depth or game.terminal_state(state)

    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -infinity
        succ = game.successors(state)
        for (a, s) in succ:
            v = max(v, min_value(s, alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        # print(f"Max at depth {depth} = {v} / alpha = {alpha}")
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = infinity
        succ = game.successors(state)
        for (a, s) in succ:
            v = min(v, max_value(s, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        # print(f"Min at depth {depth} = {v} / alpha = {alpha}")
        return v

    # Body of alphabeta_search starts here:
    succ = game.successors(state)
    action, state = argmax(succ, lambda action_state: min_value(action_state[1], -infinity, infinity, 1))
    return action


def our_best_move(board, game):
    return alphabeta_search(board, game)


def oracle_move(board, game):
    return game.oracle(board)


def make_move(board, game, opponent=chess.BLACK):
    if board.turn == opponent:
        return oracle_move(board, game)
    else:
        return our_best_move(board, game)


def main(opponent=chess.BLACK):
    try:
        engine = chess.engine.SimpleEngine.popen_uci('Stockfish/src/stockfish')
        print(f"Opened engine {engine.id.get('name')}")

        game = Game(engine, opponent=opponent)
        board = chess.variant.RacingKingsBoard()

        import pdb; pdb.set_trace()
        while True:
            m = make_move(board, game, opponent)
            t = "WHITE" if board.turn else "BLACK"
            print(f"Move: {t} {m}")
            board.push(m)
            print(board)
            if board.is_game_over():
                break

        print(f"Game over: {board.result()} "
              f"({'stockfish' if opponent==chess.WHITE else 'agent'} v {'stockfish' if opponent==chess.BLACK else 'agent'})")

    finally:
        engine.quit()


if __name__ == '__main__':
    main()

