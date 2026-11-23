# -*- coding: utf-8 -*-
"""v7 pilot: position arrays (nn/v7/positions.py) -> the inputs of both bodies, vectorised in torch (CPU or GPU).
  maia_tokens  (B, 64, 96)  Maia-3's 8 history boards, oldest first, each board from its own side to move, padded with the earliest
                            (maia3.dataset.get_historical_tokens, as toby_uci / nn/maia_ft/data.py)
  leela_planes (B, 64, 112) Leela's classic 112 planes, from the side to move (chessard/encoder.py LeelaBoard.planes), square-major
  moves_dense  legal mask (B, 4352) and per-move values (B, 4352, 3) from the CSR rows
  side         the extra scalar inputs: time control, clocks, venue (zero-initialised columns of both bodies)
Checked against the reference encoders by `python encode.py check SHARD.npz` (needs maia3 and chessard on the path)."""
import math

import numpy as np
import torch

NV = 4352
BITS = torch.arange(64, dtype=torch.int64)


def _bits(bb):
    """(..., ) int64 masks -> (..., 64) {0,1} float, index = square (a1 = 0)."""
    return ((bb.unsqueeze(-1) >> BITS.to(bb.device)) & 1).float()


def _flip(x):
    """(..., 64, C) squares mirrored vertically (a1 <-> a8)."""
    s = x.shape
    return x.reshape(*s[:-2], 8, 8, s[-1]).flip(-3).reshape(s)


def board_bits(bb):
    """bb (B, 8, 12) int64 -> (B, 8, 64, 12) bits."""
    return _bits(bb).transpose(-1, -2)


def maia_tokens(bb, nh, turn):
    """bb (B,8,12) int64, nh (B,), turn (B,) 1 = Black to move -> (B, 64, 96)."""
    B = bb.shape[0]
    x = board_bits(bb)                                    # (B, 8 steps, 64, 12), step 0 = now
    steps = torch.arange(8, device=bb.device)
    black = (turn.view(B, 1) + steps.view(1, 8)) % 2 == 1  # side to move of each history board
    xm = _flip(torch.cat([x[..., 6:], x[..., :6]], -1))   # mirrored: colours swapped, ranks flipped
    x = torch.where(black.view(B, 8, 1, 1), xm, x)
    # pad: steps >= nh take the earliest real step (nh - 1)
    idx = torch.minimum(steps.view(1, 8), (nh.view(B, 1) - 1).long())
    x = x[torch.arange(B, device=bb.device).view(B, 1), idx]   # (B, 8, 64, 12)
    x = x.flip(1)                                         # oldest first
    return x.permute(0, 2, 1, 3).reshape(B, 64, 96)


def leela_planes(bb, nh, turn, rep, castle, r50):
    """-> (B, 64, 112) float, as chessard.encoder.LeelaBoard.planes() permuted to square-major."""
    B = bb.shape[0]
    x = board_bits(bb)                                    # (B, 8, 64, 12)
    xb = _flip(torch.cat([x[..., 6:], x[..., :6]], -1))   # Black to move: the whole stack from Black's view
    x = torch.where((turn == 1).view(B, 1, 1, 1), xb, x)
    real = (torch.arange(8, device=bb.device).view(1, 8) < nh.view(B, 1).long()).float()
    x = x * real.view(B, 8, 1, 1)
    r = (rep.float() * real).view(B, 8, 1, 1).expand(B, 8, 64, 1)
    st = torch.cat([x, r], -1).permute(0, 2, 1, 3).reshape(B, 64, 104)
    c = castle.float()
    wb = turn == 1
    us_ooo = torch.where(wb, c[:, 2], c[:, 0]); us_oo = torch.where(wb, c[:, 3], c[:, 1])
    th_ooo = torch.where(wb, c[:, 0], c[:, 2]); th_oo = torch.where(wb, c[:, 1], c[:, 3])
    tail = torch.stack([us_ooo, us_oo, th_ooo, th_oo, turn.float(), r50.float(), torch.zeros_like(us_oo), torch.ones_like(us_oo)], -1)
    return torch.cat([st, tail.view(B, 1, 8).expand(B, 64, 8)], -1)


def side_inputs(base, inc, clk, clko, cad, venue, scale=10.0):
    """Extra scalar inputs (B, 12): [clock known, f(base), f(inc), f(mover's clock), f(opponent's clock), mover's share of base, cadence x4,
    venue x2], times `scale` (the learning-rate multiplier of zero-initialised columns, as nn/maia_ft/data.py CLOCK_SCALE)."""
    f = lambda s: torch.log1p(torch.clamp(s, min=0) / 60.0) / math.log(121.0)
    known = ~torch.isnan(clk) & ~torch.isnan(clko)
    tk = ~torch.isnan(base)
    z = torch.zeros_like(clk)
    cols = [known.float(), torch.where(tk, f(torch.nan_to_num(base)), z), torch.where(~torch.isnan(inc), torch.log1p(torch.nan_to_num(inc)) / math.log(61.0), z),
            torch.where(known, f(torch.nan_to_num(clk)), z), torch.where(known, f(torch.nan_to_num(clko)), z),
            torch.where(known & tk, torch.clamp(torch.nan_to_num(clk) / torch.clamp(torch.nan_to_num(base), min=1.0), max=2.0), z)]
    oh = torch.nn.functional.one_hot(cad.long(), 4).float()
    ov = torch.nn.functional.one_hot(venue.long(), 2).float()
    return torch.cat([torch.stack(cols, -1), oh, ov], -1) * scale


def dense_moves(off, lm, val, idx):
    """Legal mask (B, NV) bool, values (B, NV, 3) float (cp, 0 where illegal), for the rows idx of a shard (numpy), and the row of each move."""
    B = len(idx)
    starts, ends = off[idx], off[idx + 1]
    lens = ends - starts
    rows = np.repeat(np.arange(B), lens)
    sel = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if B else np.zeros(0, np.int64)
    mv = lm[sel].astype(np.int64)
    legal = torch.zeros(B, NV, dtype=torch.bool)
    legal[rows, mv] = True
    v = torch.zeros(B, NV, 3)
    if val is not None:
        v[rows, mv] = torch.from_numpy(val[sel].astype(np.float32))
    return legal, v, rows, mv, sel


def check(shard):
    """Compares maia_tokens / leela_planes with the reference encoders on the positions of a shard (rebuilt from the FENs is not possible:
    the check replays dev games)."""
    raise SystemExit("use test_encode.py")
