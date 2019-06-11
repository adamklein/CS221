from collections import Counter
import time
import multiprocessing

import chess
import chess.engine
import chess.variant

import numpy as np
import pandas as pd


def feature_extractor(fen, board):
    # 22 features 
    # inspired by: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.810&rep=rep1&type=pdf
    fen = Counter(fen)

    wk_rank = chess.square_rank(board.pieces(chess.KING, chess.WHITE).pop())
    bk_rank = chess.square_rank(board.pieces(chess.KING, chess.BLACK).pop())

    w_attacks = b_attacks = 0
    for square in chess.SQUARES:
        w_attacks += board.is_attacked_by(chess.WHITE, square)
        b_attacks += board.is_attacked_by(chess.BLACK, square)

    moves = {'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0,
             'k': 0, 'q': 0, 'r': 0, 'b': 0, 'n': 0}

    for mv in board.legal_moves:
        moves[board.piece_at(mv.from_square).symbol()] += 1

    board.push(chess.Move.null())

    for mv in board.legal_moves:
        moves[board.piece_at(mv.from_square).symbol()] += 1

    board.pop()

    # utility from perspective of which player it is
    features = {'wk_rank': wk_rank,
                'bk_rank': bk_rank,
                'w_attacks': w_attacks,
                'b_attacks': b_attacks}

    features.update({f'count_{k}': fen[k] for k in ['Q', 'R', 'B', 'N', 'q', 'r', 'b', 'n']})
    features.update({f'moves_{k}': v for k, v in moves.items()})

    return features


def feature_extractor2(fen, board):
    #  649 features per writeup
    fen = Counter(fen)

    names = []
    features = []

    # 10 pieces x 64-square planes => 640 features
    for color in [chess.WHITE, chess.BLACK]:
        for kind in [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            pieces = board.pieces(kind, color)
            for i, b in enumerate(pieces.tolist()):
                names.append(f'{chess.Piece(kind, color).symbol()}{i}')
                features.append(int(b))

    names.append('white_turn')
    features.append(board.turn)

    piece_count = {
        'q_count': fen['q'],
        'r_count': fen['r'],
        'b_count': fen['b'],
        'n_count': fen['n'],
        'Q_count': fen['Q'],
        'R_count': fen['R'],
        'B_count': fen['B'],
        'N_count': fen['N']}

    return dict(piece_count, **dict(zip(names, features)))


def feature_weights():
    # from linear regression approach
    # 22 features inspired by:
    #          http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.810&rep=rep1&type=pdf
    return \
        {'b_attacks': 0.009601692620996893,
         'bk_rank': -0.13841815286138437,
         'count_B': 0.20303127870591481,
         'count_N': 0.093344176471683832,
         'count_Q': 0.44136383649695127,
         'count_R': 0.26715678005335414,
         'count_b': -0.16015335773695011,
         'count_n': -0.12623773592255752,
         'count_q': -0.4019623289215134,
         'count_r': -0.24373006503474887,
         'moves_B': 0.008938157362843889,
         'moves_K': 0.019390646092190936,
         'moves_N': 0.011602012889901309,
         'moves_Q': 0.010685936601434,
         'moves_R': 0.0086658371638112172,
         'moves_b': -0.011930556222516143,
         'moves_k': -0.013206096199265072,
         'moves_n': -0.00424044971074452,
         'moves_q': -0.0092211135450850409,
         'moves_r': -0.01067582018645712,
         'w_attacks': -0.0085226705566840335,
         'wk_rank': 0.13147914925733256}


def write_feature_set(feature_file, obs_file, extractor, from_line=0, to_line=-1):
    #engine = chess.engine.SimpleEngine.popen_uci('../Stockfish/src/stockfish')

    t0 = time.time()
    with open(obs_file, 'r') as f:
        X, y = [], []
        avg = 0.100
        c = 0
        lines = list(f.readlines())
        to_line = to_line if to_line > 0 else len(lines)
        for i, l in enumerate(lines):
            if i == 0:
                continue
            if i < from_line:
                continue
            if i % 100 == 0:
                avg = c / (time.time() - t0)
                if avg != 0:
                    print(f'Progress for {feature_file}: line {i} read: estimate remaining time {int((to_line-i)/avg)}s')
            if i >= to_line:
                break
            fen, score = l.split(',')
            score = float(score)
            board = chess.variant.RacingKingsBoard(fen)
            # score = float(engine.analyse(board, chess.engine.Limit(time=0.100))['score'].white().score(mate_score=1000))
            X.append(extractor(fen, board))
            y.append(score)
            c += 1

        f = pd.DataFrame.from_records(X)
        f['score'] = np.array(y)
        f.to_parquet(feature_file, index=False)


if __name__ == '__main__':
    raw = 'data/racingkings/racing_king_train.csv'
    feat = 'data/nn_orig_features_train{}.parquet'

    write_feature_set(feat.format(''), raw, feature_extractor2, 0)

    # # parallelized feature extraction:
    # procs = []
    # for i in range(10):
    #     chunk = 35000
    #     s = i * chunk
    #     e = (i+1) * chunk
    #     p = multiprocessing.Process(target=write_feature_set, args=(feat.format(i), obs, feature_extractor2, s, e))
    #     p.start()
    #     procs.append(p)
    #
    # for p in procs:
    #    p.join()

