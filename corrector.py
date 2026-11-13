# -*- coding: utf-8 -*-
"""Neural corrector of Maia's move probabilities with Leela's evaluations (replaces the hand-made rule of toby_uci.toby_distribution).

One function builds the inputs (`build_inputs`), the same for training (nn/train.py) and for the engine (toby_uci.py), so that what the network
sees at play time is exactly what it saw in training. The forward pass is plain numpy (no torch needed by the engine for this part).

Model: for each legal move m,   score(m) = log p_Maia(m) + f(context, move features of m),   p_Toby = softmax(score) over the legal moves.
f is a small MLP whose last layer starts at zero, so an untrained network gives exactly Maia's distribution.
Expected scores (ES) are always used instead of raw evaluations (CLAUDE.md): Leela's Q is converted to ES = (1 + Q) / 2, and also to centipawns
(lc0's own Q -> cp formula) then to the rating-dependent ES(R, cp) of the project (find_maia_blunders.expected_score).
"""
import json
import math

import numpy as np

CADENCES = ("classical", "rapid", "blitz")
MIN_N = 8   # as in toby_uci: a move needs this many visits for its main-search value to be trusted
CTX_NAMES = ["x", "E", "classical", "rapid", "blitz", "online", "opp_known", "opp_diff", "root_es", "root_d", "maia_entropy", "maia_logmax",
             "unvalued_mass", "best_gap", "log_legal", "ply", "material", "in_check", "has_maia2",
             "has_history", "has_aux"]   # added for the history / auxiliary model versions (a network only reads the inputs listed in its weights file)
MOVE_NAMES = ["lp", "lp2_minus_lp", "valued", "d_es", "loss_r", "is_best", "log_n", "n_frac", "log_pol", "from_complement", "maia_rank",
              "capture", "check", "promotion", "draw", "es_abs",
              "lp_aux_minus_lp", "aux_rank",
              "sf_budget",   # Stockfish version: log10(node budget) - 5.5
              "sf_shallow_d_es", "sf_shallow_es",
              # Maia's value head: after each plausible move (mover's expected score - 0.5, gap to the best valued move, known) and of the position
              "mval_es", "mval_gap", "mval_known", "mroot_es", "mroot_draw",
              # PLAN_RETRAIN_V6.md section 10 (5 November): opening table, safe moves, V2 instability, move number (position-level values repeated per move)
              "tab_logn", "tab_share", "n_safe5", "v2_best_changed", "v2_d_es", "v2_ext", "move_no",
              # stage 3 (F1g): level interactions, lvl = max(0, E - 2400) / 400 (nn/retrain_v6/add_lvl.py)
              "lvl_loss", "lvl_small", "lvl_inacc"]   # Stockfish version: the same search at a shallow depth (expected-score gap to that depth's best move,
                                                    # and the move's expected score - 0.5); a move that looks good early but not at the end, or the reverse
# (the Stockfish names are kept last so that older data files keep their columns)
N_CTX, N_MOVE = len(CTX_NAMES), len(MOVE_NAMES)
PIECE_VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def q_to_cp(q):
    """lc0's conversion of Q (in [-1, 1]) to centipawns."""
    q = max(-0.9999, min(0.9999, q))
    return 111.714640912 * math.tan(1.5620688421 * q)


def expected_score(rating, cp):
    """ES(R, E) of CLAUDE.md (find_maia_blunders.expected_score), in [0, 1]; normal scores capped at +-2000 cp."""
    e = max(-2000.0, min(2000.0, cp))
    return 0.5 + 0.5 * (2 / (1 + math.exp((rating * -0.00000274 + 0.00048) * e)) - 1)


def move_value(u, lee, extra):
    """Leela's Q of a move for the mover, with the engine's rule: main search if >= MIN_N visits, else the 64-node complement, else the few visits; None if unknown.
    Returns (Q, source) with source 0 = main search, 1 = complement, 2 = few visits."""
    v = lee.get(u)
    if v and v[0] >= MIN_N:
        return v[3], 0
    if u in extra:
        return extra[u], 1
    if v and v[0] >= 1:
        return v[3], 2
    return None, None


def sf_pseudo_leela(sf, wdl=None, wl_from_wdl=False):
    """Stockfish's evaluation of every legal move ({uci: centipawns for the side to move, mates +-10000}) turned into Leela-like values, the inputs
    of the Stockfish version of Toby: Q = the inverse of lc0's Q -> centipawn formula (so that q_to_cp gives the centipawns back), the same 'visits'
    for every move, the draw value from Stockfish's win/draw/loss when wdl is given (else 0); wl_from_wdl: Q = win - loss of that
    win/draw/loss instead of the centipawns. Returns ({uci: (N, WL, D, Q)}, root (WL, D) = the best move's value)."""
    lee = {}
    for u, cp in sf.items():
        q = max(-0.999, min(0.999, math.atan(cp / 111.714640912) / 1.5620688421))
        dr = (wdl[u][1] / 1000.0) if (wdl and u in wdl) else 0.0   # Stockfish's own draw probability of the move (UCI_ShowWDL), when given
        if wl_from_wdl and wdl and u in wdl:   # win - loss of Stockfish's win/draw/loss instead of the centipawns
            q = max(-0.999, min(0.999, (wdl[u][0] - wdl[u][2]) / 1000.0))
        lee[u] = (100, q, dr, q)
    if not lee:
        return lee, None
    best = max(lee, key=lambda u: lee[u][3])
    return lee, (lee[best][3], lee[best][2])


def build_inputs(board, pm, lee, pol, extra, root, x, E, cadence, venue, E_opp=None, pm2=None, history=False, pm_aux=None, sf_budget=None,
                 sf_shallow=None, maia_vals=None, maia_root=None, opening=None, v2info=None):
    """Inputs of the network for one position.
    board: chess.Board (side to move = the player); pm: Maia's probabilities {uci: p} at the Maia input rating x (same rating on both sides);
    lee: Leela's main search {uci: (N, WL, D, Q)}; pol: Leela's policy {uci: P}; extra: 64-node complements {uci: Q}; root: (WL, D) of the root;
    E: FIDE-equivalent strength of the player; cadence: classical / rapid / blitz; venue: otb / online; E_opp: opponent's strength or None;
    pm2: optional second Maia distribution (at the player's own rating), or None; history: Maia (and auxiliary model) received the previous positions;
    pm_aux: unused (kept for old weights files), None; sf_budget: Stockfish node budget of the values (Stockfish version), or None.
    Returns (moves, ctx [N_CTX], mv [n_moves, N_MOVE], base [n_moves] = log p_Maia)."""
    moves = list(pm)
    n = len(moves)
    p = np.array([max(pm[u], 1e-6) for u in moves])
    lp = np.log(p)
    best = max(lee, key=lambda u: (lee[u][0], lee[u][3])) if lee else None
    qs = [move_value(u, lee, extra) for u in moves]
    qb = move_value(best, lee, extra)[0] if best else None
    tot_n = max(1, sum(v[0] for v in lee.values()))
    es_b = (1 + qb) / 2 if qb is not None else None
    esr_b = expected_score(E, q_to_cp(qb)) if qb is not None else None
    mv = np.zeros((n, N_MOVE))
    order = np.argsort(-p)
    rank = np.empty(n)
    rank[order] = np.arange(n)
    valued_es = []
    unvalued = 0.0
    for i, u in enumerate(moves):
        q, src = qs[i]
        m = board.parse_uci(u)
        mv[i, 0] = lp[i]
        if pm2 is not None:
            mv[i, 1] = math.log(max(pm2.get(u, 1e-6), 1e-6)) - lp[i]
        if q is not None and qb is not None:
            es = (1 + q) / 2
            mv[i, 2] = 1.0
            mv[i, 3] = max(-1.0, min(1.0, es - es_b))
            mv[i, 4] = max(-1.0, min(1.0, esr_b - expected_score(E, q_to_cp(q))))
            mv[i, 9] = 1.0 if src == 1 else 0.0
            mv[i, 15] = es - 0.5
            valued_es.append(es)
        else:
            unvalued += pm[u]
        v = lee.get(u)
        nv = v[0] if v else 0
        mv[i, 5] = 1.0 if u == best else 0.0
        mv[i, 6] = math.log1p(nv) / 7
        mv[i, 7] = nv / tot_n
        mv[i, 8] = math.log(max(pol.get(u, 0.0), 1e-4)) / 5 if pol else 0.0
        mv[i, 10] = math.log1p(rank[i]) / 3
        mv[i, 11] = 1.0 if board.is_capture(m) else 0.0
        mv[i, 12] = 1.0 if board.gives_check(m) else 0.0
        mv[i, 13] = 1.0 if m.promotion else 0.0
        mv[i, 14] = v[2] if (v and nv >= 1) else 0.0
        if pm_aux is not None:
            mv[i, 16] = max(-8.0, min(8.0, math.log(max(pm_aux.get(u, 1e-6), 1e-6)) - lp[i]))
    if pm_aux is not None:
        pd = np.array([pm_aux.get(u, 0.0) for u in moves])
        drank = np.empty(n)
        drank[np.argsort(-pd)] = np.arange(n)
        mv[:, 17] = np.log1p(drank) / 3
    if sf_budget:
        mv[:, 18] = math.log10(sf_budget) - 5.5
    if sf_shallow:   # {uci: centipawns} of the same Stockfish search at a shallow depth
        qsh = {u: max(-0.999, min(0.999, math.atan(cp / 111.714640912) / 1.5620688421)) for u, cp in sf_shallow.items()}
        esb = (1 + max(qsh.values())) / 2
        for i, u in enumerate(moves):
            if u in qsh:
                es = (1 + qsh[u]) / 2
                mv[i, 19] = max(-1.0, min(1.0, es - esb))
                mv[i, 20] = es - 0.5
    if maia_vals:   # {uci: [loss, draw, win] of the position after the move, for the side then to move}: the mover's expected score
        mes = {u: v[0] + v[1] / 2 for u, v in maia_vals.items()}
        mb = max(mes.values())
        for i, u in enumerate(moves):
            if u in mes:
                mv[i, 21] = mes[u] - 0.5
                mv[i, 22] = mes[u] - mb
                mv[i, 23] = 1.0
    if maia_root:   # [loss, draw, win] of the position for the side to move
        mv[:, 24] = maia_root[2] + maia_root[1] / 2 - 0.5
        mv[:, 25] = maia_root[1]
    # section 10 inputs. opening: the elite opening table {fen4: [n, {uci: count}]} or None; v2info: (best move of the 150k base search, best
    # move after the 250k extension, {uci: (base cp, extension cp)} of the extended candidates) or None. n_safe5 and move_no are always set.
    cnt = opening.get(" ".join(board.fen().split()[:4])) if opening else None
    if cnt:
        mv[:, 26] = math.log1p(cnt[0]) / 5
        for i, u in enumerate(moves):
            mv[i, 27] = cnt[1].get(u, 0) / cnt[0]
    mv[:, 28] = math.log1p(int(((mv[:, 2] > 0.5) & (mv[:, 4] <= 0.05)).sum())) / 2
    if v2info:
        b0, b1, cands = v2info
        mv[:, 29] = 1.0 if b0 != b1 else 0.0
        for i, u in enumerate(moves):
            if u in cands:
                c0, c1 = cands[u]
                mv[i, 30] = max(-1.0, min(1.0, expected_score(E, c1) - expected_score(E, c0)))
                mv[i, 31] = 1.0
    mv[:, 32] = min(board.fullmove_number, 80) / 40
    lvl = max(0.0, E - 2400) / 400   # F1g's level inputs, as nn/retrain_v6/add_lvl.py (loss_r = column 4)
    mv[:, 33] = mv[:, 4] * lvl
    mv[:, 34] = ((mv[:, 4] > 0.005) & (mv[:, 4] <= 0.05)) * lvl
    mv[:, 35] = ((mv[:, 4] > 0.05) & (mv[:, 4] <= 0.10)) * lvl
    ent = float(-(p / p.sum() * np.log(p / p.sum())).sum())
    srt = sorted(valued_es, reverse=True)
    gap = (srt[0] - srt[1]) if len(srt) >= 2 else 0.0
    mat = sum(PIECE_VAL[pc.piece_type] for pc in board.piece_map().values())
    ctx = np.zeros(N_CTX)
    ctx[0] = (x - 1800) / 600
    ctx[1] = (E - 2000) / 500
    ctx[2 + CADENCES.index(cadence)] = 1.0
    ctx[5] = 1.0 if venue == "online" else 0.0
    if E_opp is not None:
        ctx[6] = 1.0
        ctx[7] = max(-1.25, min(1.25, (E - E_opp) / 400))
    ctx[8] = (root[0] / 2) if root else 0.0
    ctx[9] = root[1] if root else 0.0
    ctx[10] = ent / 3
    ctx[11] = float(lp.max())
    ctx[12] = unvalued
    ctx[13] = gap
    ctx[14] = math.log(max(n, 1)) / 4
    ctx[15] = 0.0   # move number: always 0 in training (boards rebuilt from FENs), so 0 in the engine too (measured effect of the old value: 0.15 point of probability)
    ctx[16] = mat / 78
    ctx[17] = 1.0 if board.is_check() else 0.0
    ctx[18] = 1.0 if pm2 is not None else 0.0
    ctx[19] = 1.0 if history else 0.0
    ctx[20] = 1.0 if pm_aux is not None else 0.0
    return moves, ctx, mv, lp


class Corrector:
    """Numpy forward pass of a trained network (weights saved by nn/train.py)."""

    def __init__(self, path):
        d = json.loads(open(path, encoding="utf-8").read())
        self.meta = d["meta"]
        self.mu = np.array(d["mu"])
        self.sd = np.array(d["sd"])
        self.layers = [(np.array(W), np.array(b)) for W, b in d["layers"]]
        names = self.meta.get("inputs") or (CTX_NAMES[:19] + MOVE_NAMES[:16])   # the first network (v1) read 19 + 16 inputs
        full = CTX_NAMES + MOVE_NAMES
        self.cols = np.array([full.index(nm) for nm in names])

    def scores(self, ctx, mv, base):
        n = mv.shape[0]
        h = np.concatenate([np.repeat(ctx[None, :], n, axis=0), mv], axis=1)[:, self.cols]
        h = (h - self.mu) / self.sd
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < len(self.layers) - 1:
                h = h / (1 + np.exp(-h))   # SiLU
        return base + h[:, 0]

    def distribution(self, ctx, mv, base, moves, temperature=1.0):
        s = self.scores(ctx, mv, base) / temperature
        s = np.exp(s - s.max())
        s /= s.sum()
        return {u: float(v) for u, v in zip(moves, s)}
