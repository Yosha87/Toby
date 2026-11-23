# -*- coding: utf-8 -*-
"""v7 pilot networks: one body, a move head and a W/D/L head (nn/v7/PILOT_PLAN.md section 2). Every new input enters through
zero-initialised weights, so that at step 0 body A is exactly the released Maia-3 79M and body B exactly the lc0 BT4 network (checked by
`python train.py selfcheck`).

Per-move value features (both bodies, 7 columns, zero where an option does not use them):
  [ES9(move), ES9(best), valued, ES5(move), ES1(move), ES5(best), ES1(best)]
  ES = CLAUDE.md expected score at the mover's strength / 100 (both bodies).
  Option C: the first 3 (valued = 1); T: the same on the valued moves only (others 0, valued 0); Cmd (multi-depth): all 7.
Position summary for the W/D/L head (6 columns): [ES9(best), ES9(best) - ES9(2nd), valued share, ES5(best), ES1(best), 1].
Outputs: move logits (B, 4352) with -inf on illegal moves; W/D/L logits (B, 3) in the order (loss, draw, win) for the side to move."""
import math

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

NV = 4352
NF = 7      # per-move value features
NL = 5 if os.environ.get("V7_LEELA") == "1" else 0   # Leela search inputs of the value path (pilot): visit share, Q, visited, root WL, present
NS = 6      # position summary features
NSIDE = 12  # encode.side_inputs


def move_squares():
    t = torch.empty(NV, 2, dtype=torch.long)
    i = torch.arange(4096)
    t[:4096, 0], t[:4096, 1] = i // 64, i % 64
    j = torch.arange(256)
    t[4096:, 0], t[4096:, 1] = 48 + j // 32, 56 + (j % 32) // 4
    return t


class MoveValueHead(nn.Module):
    """Body A: MLP([from-square vector, to-square vector, value features]) -> a logit added to Maia-3's logit (last layer zero)."""

    def __init__(self, d, hid=256):
        super().__init__()
        self.fc1 = nn.Linear(2 * d + NF, hid)
        self.fc2 = nn.Linear(hid, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, a, b, f):
        return self.fc2(F.gelu(self.fc1(torch.cat([a, b, f], -1)))).squeeze(-1)


class BodyA(nn.Module):
    """Maia-3 79M + time / venue inputs (zero columns of the token projection) + per-move value head + W/D/L head (Maia's value head,
    plus the Stockfish summary through a zero-initialised layer)."""

    def __init__(self, maia):
        super().__init__()
        self.m = maia
        cfg = maia.cfg
        tp = maia.token_projection
        h = 12 * cfg.history
        assert not cfg.include_time_info and tp.in_features == h + 2 * cfg.dim_emb
        new = nn.Linear(tp.in_features + NSIDE, tp.out_features)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :h] = tp.weight[:, :h]
            new.weight[:, h + NSIDE:] = tp.weight[:, h:]
            new.bias.copy_(tp.bias)
        maia.token_projection = new
        self.h = h
        self.vhead = MoveValueHead(cfg.head_hid_dim)
        self.wdl_sf = nn.Linear(NS, cfg.head_hid_dim)
        nn.init.zeros_(self.wdl_sf.weight)
        nn.init.zeros_(self.wdl_sf.bias)
        self.register_buffer("msq", move_squares(), persistent=False)
        promo = []
        import chess
        for ff in range(8):
            for tf in range(8):
                for p in range(4):
                    promo.append((chess.square(ff, 6), chess.square(tf, 7), tf, p))
        self.register_buffer("promo", torch.tensor(promo), persistent=False)

    def new_modules(self):
        return [self.vhead, self.wdl_sf]

    def new_params(self):
        """Parameters that did not exist in Maia-3 (higher learning rate): the new heads; the new input columns are part of the token
        projection (body learning rate, as Maia C's time inputs) but scaled by encode.side_inputs."""
        return [p for m in self.new_modules() for p in m.parameters()]

    def forward(self, tokens, side, elo, elo_opp, legal, feats, summ):
        m = self.m
        cfg = m.cfg
        B = tokens.shape[0]
        se = m.interpolate_elo(elo).unsqueeze(1).expand(-1, 64, -1)
        so = m.interpolate_elo(elo_opp).unsqueeze(1).expand(-1, 64, -1)
        x = torch.cat([tokens, side.unsqueeze(1).expand(-1, 64, -1).to(tokens.dtype), se, so], -1)
        x = m.token_projection(x)
        if hasattr(m, "abs_pe"):
            x = m.abs_pe(x)
        x = m.transformer(x)
        sq_from = m.proj_sq_from(x)
        sq_to = m.proj_sq_to(x)
        base = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(cfg.head_hid_dim)
        flat = base.reshape(B, 4096)
        pb = m.promo_bias_proj(sq_to[:, 56:64, :]) * math.sqrt(cfg.head_hid_dim)   # (B, 8, 4)
        P = self.promo
        promo = base[:, P[:, 0], P[:, 1]] + pb[:, P[:, 2] - 0, P[:, 3]]
        logits = torch.cat([flat, promo], 1).float()
        bi, mi = legal.nonzero(as_tuple=True)
        sq = self.msq[mi]
        add = self.vhead(sq_from[bi, sq[:, 0]], sq_to[bi, sq[:, 1]], feats[bi, mi].to(sq_from.dtype)).float()
        logits = logits.masked_fill(~legal, float("-inf"))
        logits = logits.index_put((bi, mi), logits[bi, mi] + add)
        pooled = m.last_ln(x.mean(dim=1))
        hv = F.relu(m.fc_value_hid(pooled) + self.wdl_sf(summ.to(pooled.dtype)))
        wdl = m.fc_value(hv).float()
        return logits, wdl


RBF_C = torch.linspace(600, 2900, 12)


def rating_rbf(E):
    """(B,) strengths -> (B, 12) radial features (centres 600-2900, width 250): the rating inputs of body B."""
    return torch.exp(-((E.float().view(-1, 1) - RBF_C.to(E.device).view(1, -1)) / 250.0) ** 2)


class BodyB(nn.Module):
    """Official Leela BT4 (nn/v7/leela_torch.py, protocol amendment 1) + both ratings, time control, clocks and venue through
    zero-initialised layers (into the embedding, and again at the input of the first trainable encoder layer), Leela's attention policy
    as the move head (+ a zero-initialised per-move MLP that also reads the value features), Leela's value head as the W/D/L head
    (+ the Stockfish summary through a zero-initialised layer). Encoder layers [0, freeze) frozen (0 = full fine-tune)."""

    NX = 24 + NSIDE

    def __init__(self, lc0, freeze=0, vpath=False):
        super().__init__()
        self.lc0 = lc0
        # vpath (post-pilot amendment 4): a separate low-dimensional value path, MLP([value features, both ratings]) -> a logit added to
        # every legal move, so that the 7 value columns are not drowned next to two 1,024-wide square vectors (A's value head learnt
        # little in the pilot). Last layer zero-initialised: step 0 unchanged.
        self.vpath = None
        self.lpath = None
        if NL:   # Toby 2 Accurate: a small Leela head on a frozen B (value features, Leela search inputs, both ratings) -> a logit per move
            self.lpath = nn.Sequential(nn.Linear(NF + NL + 24, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.lpath[-1].weight)
            nn.init.zeros_(self.lpath[-1].bias)
        if vpath:
            self.vpath = nn.Sequential(nn.Linear(NF + 24, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.vpath[-1].weight)
            nn.init.zeros_(self.vpath[-1].bias)
        E = lc0.E
        self.extra_lin = nn.Linear(self.NX, E)
        self.inject = nn.Linear(self.NX, E)
        for l in (self.extra_lin, self.inject):
            nn.init.zeros_(l.weight)
            nn.init.zeros_(l.bias)
        self.vhead = MoveValueHead(E)
        self.vhead.fc1 = nn.Linear(2 * E + NF + 24, self.vhead.fc1.out_features)
        self.wdl_sf = nn.Linear(NS, lc0.v_d1.out_features)
        nn.init.zeros_(self.wdl_sf.weight)
        nn.init.zeros_(self.wdl_sf.bias)
        self.freeze = freeze
        for i, enc in enumerate(lc0.enc):
            if i < freeze:
                for p in enc.parameters():
                    p.requires_grad_(False)
        import leela_torch as LT
        vt = LT.vocab_tables()                       # (8, NV) grid index per transform
        self.register_buffer("vt", vt, persistent=False)
        sq = torch.zeros(8, NV, 2, dtype=torch.long)  # transformed (from, to) squares of each move
        ms = move_squares()
        for t in range(8):
            for k in range(NV):
                sq[t, k, 0] = LT._sq_t(int(ms[k, 0]), t)
                sq[t, k, 1] = LT._sq_t(int(ms[k, 1]), t)
        self.register_buffer("sqt", sq, persistent=False)

    def new_params(self):
        return list(self.extra_lin.parameters()) + list(self.inject.parameters()) + list(self.vhead.parameters()) + \
            list(self.wdl_sf.parameters()) + (list(self.vpath.parameters()) if self.vpath is not None else []) + \
            (list(self.lpath.parameters()) if self.lpath is not None else [])

    def forward(self, planes, side, elo, elo_opp, legal, feats, summ, transform, lfeat=None):
        rr = torch.cat([rating_rbf(elo), rating_rbf(elo_opp)], -1)
        extra = torch.cat([rr, side.to(rr.dtype)], -1)
        x = self.lc0.body(planes, extra, self.extra_lin, self.inject(extra) if self.freeze > 0 else None, self.freeze)
        grid = self.lc0.policy_grid(x)
        import leela_torch as LT
        idx = LT.castle_fix(self.vt[transform], planes)   # (B, NV)
        logits = grid.gather(1, idx).float().masked_fill(~legal, float("-inf"))
        bi, mi = legal.nonzero(as_tuple=True)
        sq = self.sqt[transform[bi], mi]   # (the per-move MLP reads the king's from / to squares of castling, as Maia's vocabulary)
        fr = torch.cat([feats[bi, mi], rr[bi]], -1)
        add = self.vhead(x[bi, sq[:, 0]], x[bi, sq[:, 1]], fr.to(x.dtype)).float()
        if self.vpath is not None:
            add = add + self.vpath(fr.float()).squeeze(-1).float()
        if self.lpath is not None and lfeat is not None:   # zero at initialisation: B unchanged until the head learns
            fl = torch.cat([feats[bi, mi], lfeat[bi, mi].to(feats.dtype), rr[bi]], -1)
            add = add + self.lpath(fl.float()).squeeze(-1).float()
        logits = logits.index_put((bi, mi), logits[bi, mi] + add)
        wdl = self.lc0.value(x, self.wdl_sf(summ.to(x.dtype))).float()[:, [2, 1, 0]]
        return logits, wdl


def value_features(vals, legal, valued, E, body, use_md):
    """vals (B, NV, 3) cp at depths (1, 5, 9) from the mover's view; valued (B, NV) bool; E (B,) mover's strength.
    -> feats (B, NV, NF), summ (B, NS) (see the module docstring)."""
    def conv(cp):   # the CLAUDE.md expected score at the mover's strength (0-1), mates +-10000 uncapped, other scores capped at 2000
        c = torch.where(cp.abs() < 9000, cp.clamp(-2000, 2000), cp)
        k = (E * -0.00000274 + 0.00048).view(-1, 1)
        return 0.5 + 0.5 * (2 / (1 + torch.exp(k * c)) - 1)
    v = valued & legal
    e9, e5, e1 = conv(vals[..., 2]), conv(vals[..., 1]), conv(vals[..., 0])
    neg = torch.tensor(-1.0, device=vals.device)
    def best(e):
        return torch.where(v, e, neg).max(1).values.clamp(min=0)
    b9, b5, b1 = best(e9), best(e5), best(e1)
    top2 = torch.where(v, e9, neg).topk(2, 1).values
    gap = torch.where(top2[:, 1] >= 0, top2[:, 0] - top2[:, 1], torch.zeros_like(b9))
    vf = v.float()
    unval = 0.0
    m9 = torch.where(v, e9, torch.full_like(e9, unval))
    z = torch.zeros_like(e9)
    md = 1.0 if use_md else 0.0
    if use_md is None:   # option "none": no Stockfish values (the human-learning test, amendment 3)
        return torch.zeros(*vals.shape[:2], NF, device=vals.device), torch.zeros(vals.shape[0], NS, device=vals.device)
    feats = torch.stack([m9, b9.view(-1, 1).expand_as(e9), vf, torch.where(v, e5, z) * md, torch.where(v, e1, z) * md,
                         (b5 * md).view(-1, 1).expand_as(e9), (b1 * md).view(-1, 1).expand_as(e9)], -1)
    share = vf.sum(1) / legal.float().sum(1).clamp(min=1)
    summ = torch.stack([b9, gap, share, b5 * md, b1 * md, torch.ones_like(b9)], -1)
    return feats, summ
