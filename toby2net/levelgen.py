# -*- coding: utf-8 -*-
"""Toby 2 level head, generative form (v1b): a rating-conditioned move model on ONE pass of Toby 2's body.

Body B runs once per half-move at mover E = opponent Eo = E_BODY (2000), the game's cadence, venue otb, clocks unknown, the history
(levelhead.py: B's ratings are inputs, so they are fixed; no rating information about the game enters).

Move term. For every legal move m, B's values-masked move logit at 2000, base(m) (exactly Toby 2's head at E = Eo = 2000), plus a
learned rating-conditioned correction:
    logit(m | E) = base(m) + a(m) + u(m) . v(E)
u(m), a(m) from B's tokens at m's from / to squares and the mean token (as B's per-move MLP reads them); v(E) from B's own rating RBF.
log p(played | position, E) = log_softmax_m(logit(. | E))[played]: a likelihood of the move given the position (as Toby 1.0's and Toby
2's), so a player's half-moves are SUMMED. Trained by cross-entropy at the mover's real rating; B is frozen.

Loss term (when Stockfish values are given). The expected-score loss of the played move at R = 1500 (CLAUDE.md formula: mates +-10000
uncapped, other scores capped at +-2000; never raw centipawns), put in LOSS_BINS, modelled generatively:
    log p(bin | position context, E), context = the mean token, ES(1500, best), ply, cadence, log(legal moves), B's entropy at 2000.
Without values the term is 0. The per-move curve over GRID (1200-3000 step 25) = move term (+ loss term)."""
import math

import numpy as np

import levelhead as LH

E_BODY = LH.E_BODY
GRID = LH.GRID
LOSS_BINS = (0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 40.0)   # bin k: loss in (edge[k-1], edge[k]]; bin 0: exactly 0; last: > 40 ES points
FORMAT = "toby2-levelgen/1"
N_CTXS = 7   # ES(1500, best) / 100, ply / 100, cadence x3, log(legal), entropy at 2000


def loss_bin(best_cp, played_cp):
    """-> bin index of ES(1500, best) - ES(1500, played) (points), or -1 if values are missing; also ES(1500, best) / 100."""
    if best_cp is None or played_cp is None:
        return -1, 0.0
    b, p = LH.score_to_cp(best_cp), LH.score_to_cp(played_cp)
    b = max(b, p)
    eb = LH.expected_score(1500.0, b)
    loss = eb - LH.expected_score(1500.0, p)
    k = 0 if loss <= 0 else 1 + int(np.searchsorted(LOSS_BINS[1:], loss, side="left"))
    return min(k, len(LOSS_BINS)), eb / 100.0


def loss_bin_torch(best, played):
    import torch
    miss = torch.isnan(best) | torch.isnan(played)
    cap = lambda v: torch.where(v.abs() >= 10000, torch.sign(v) * 10000.0, v.clamp(-2000.0, 2000.0))
    b, p = cap(torch.nan_to_num(best)), cap(torch.nan_to_num(played))
    b = torch.maximum(b, p)
    es = lambda v: 100 * (0.5 + 0.5 * (2 / (1 + torch.exp((1500.0 * -0.00000274 + 0.00048) * v)) - 1))
    loss = es(b) - es(p)
    edges = torch.tensor(LOSS_BINS[1:], device=best.device)
    k = torch.where(loss <= 0, torch.zeros_like(loss, dtype=torch.long), 1 + torch.searchsorted(edges, loss.contiguous(), right=False))
    k = k.clamp(max=len(LOSS_BINS))
    return torch.where(miss, torch.full_like(k, -1), k), es(b) / 100.0


def make_gen(du=64, hidden=256, arch="bilinear", n_anchor=1):
    """arch "bilinear": logit = base + a(m) + u(m) . v(E); "mlp": logit = base + MLP(GELU(h(m) + e(E))), a full interaction of the
    move's embedding and the rating's. n_anchor > 1: B's passes at several fixed anchor ratings (inputs concatenated; the anchors' base
    logits mixed with weights that depend on E)."""
    import torch
    import torch.nn as nn

    class LevelGen(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            if arch == "mlp":
                self.h = nn.Sequential(nn.Linear(3 * 1024 * n_anchor, hidden), nn.LayerNorm(hidden))
                self.e = nn.Sequential(nn.Linear(12, 64), nn.GELU(), nn.Linear(64, hidden))
                self.o = nn.Sequential(nn.GELU(), nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1))
                nn.init.zeros_(self.o[-1].weight)
                nn.init.zeros_(self.o[-1].bias)
            self.n_anchor = n_anchor
            self.w = nn.Linear(12, n_anchor)
            nn.init.zeros_(self.w.weight)
            nn.init.zeros_(self.w.bias)
            self.u = nn.Sequential(nn.Linear(3 * 1024 * n_anchor, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, du + 1))
            self.v = nn.Sequential(nn.Linear(12, 64), nn.GELU(), nn.Linear(64, du))
            nn.init.zeros_(self.u[-1].weight)
            nn.init.zeros_(self.u[-1].bias)
            self.cx = nn.Sequential(nn.Linear(1024 * n_anchor, 64), nn.LayerNorm(64))
            self.lm = nn.Sequential(nn.Linear(64 + N_CTXS + 32, 128), nn.GELU(), nn.Linear(128, len(LOSS_BINS) + 1))
            self.v2 = nn.Sequential(nn.Linear(12, 32), nn.GELU())

        def move_logits(self, base, xf, xt, xm, rbf):
            """base (B, L) or (B, L, A) anchors, xf / xt (B, L, 1024 A), xm (B, 1024 A), rbf (B, K, 12) -> (B, K, L) logits."""
            if base.dim() == 2:
                base = base.unsqueeze(-1)
            wts = torch.softmax(self.w(rbf.float()), -1)                                     # (B, K, A)
            base = torch.einsum("bla,bka->bkl", base.nan_to_num(neginf=-1e9), wts).masked_fill(torch.isinf(base[..., 0]).unsqueeze(1), float("-inf"))
            if self.arch == "mlp":
                h = self.h(torch.cat([xf, xt, xm.unsqueeze(1).expand_as(xf)], -1).float())   # (B, L, H)
                e = self.e(rbf.float())                                                       # (B, K, H)
                return base + self.o(h.unsqueeze(1) + e.unsqueeze(2))[..., 0]
            h = self.u(torch.cat([xf, xt, xm.unsqueeze(1).expand_as(xf)], -1).float())   # (B, L, du + 1)
            a, u = h[..., 0], h[..., 1:]
            v = self.v(rbf.float())                                                       # (B, K, du)
            return base + a.unsqueeze(1) + torch.einsum("bld,bkd->bkl", u, v)

        def loss_logits(self, xm, ctx, rbf):
            """xm (B, 1024), ctx (B, N_CTXS), rbf (B, K, 12) -> (B, K, n_bins) log-probabilities."""
            K = rbf.shape[1]
            c = torch.cat([self.cx(xm.float()), ctx.float()], -1).unsqueeze(1).expand(-1, K, -1)
            return torch.log_softmax(self.lm(torch.cat([c, self.v2(rbf.float())], -1)), -1)

    return LevelGen()


class Base:
    """B's body at E = Eo = E_BODY, batched over positions, and B's masked move logits at 2000 for every legal move."""

    def __init__(self, net):
        import torch
        import toby2 as T2
        self.torch, self.T2, self.tn = torch, T2, net
        self.M = net.M
        self.dev = net.device
        self.dt = torch.float16 if net.half else torch.float32
        self.rr0 = torch.cat([self.M.rating_rbf(torch.tensor([E_BODY])), self.M.rating_rbf(torch.tensor([E_BODY]))], -1)
        self.sides = {c: net.side_inputs(None, None, None, None, c, "otb") for c in LH.CADS}
        self.rbf_grid = self.M.rating_rbf(torch.tensor(GRID, dtype=torch.float32))   # (73, 12)

    def indices(self, planes, transform, mi):
        """Policy-grid entries and transformed from / to squares of the legal moves (CPU)."""
        net = self.tn.net
        idx = self.T2.LT.castle_fix(net.vt[transform:transform + 1].cpu(), planes)[0, mi]
        sq = net.sqt[transform].cpu()[mi]
        return idx, sq

    def run(self, P, cad_idx, idx_pad, sq_pad, mask, E=None):
        """P (B, 112, 8, 8), cad_idx (B,) index in LH.CADS, idx_pad (B, L) policy-grid entries, sq_pad (B, L, 2), mask (B, L) legal ->
        base (B, L) masked logits at 2000 (-inf on padding), xf, xt (B, L, 1024), xm (B, 1024), entropy (B,).
        E (B,) given: B at mover = opponent = E per row instead of 2000 (Toby 2's own masked policy at E: the distillation teacher;
        the value path's term is the same for every move of a row, so it is left out of the logits)."""
        torch, T2 = self.torch, self.T2
        net = self.tn.net
        n = P.shape[0]
        with torch.inference_mode():
            side = torch.cat([self.sides[LH.CADS[int(c)]] for c in cad_idx], 0).to(self.rr0.dtype)
            if E is None:
                rr = self.rr0.expand(n, -1)
            else:
                r1 = self.M.rating_rbf(E.float().cpu())
                rr = torch.cat([r1, r1], -1)
            extra = torch.cat([rr, side], -1).to(self.dev, self.dt)
            x = net.lc0.body(P.to(self.dev, self.dt), extra, net.extra_lin, net.inject(extra) if net.freeze > 0 else None, net.freeze)
            grid = net.lc0.policy_grid(x)                                              # (B, 4288)
            idx_pad, sq_pad, mask = idx_pad.to(self.dev), sq_pad.to(self.dev), mask.to(self.dev)
            lg = torch.gather(grid, 1, idx_pad).float()
            bi = torch.arange(n, device=self.dev).view(-1, 1).expand_as(idx_pad)
            xf, xt = x[bi, sq_pad[..., 0]], x[bi, sq_pad[..., 1]]                      # (B, L, 1024)
            L = idx_pad.shape[1]
            fr = torch.cat([torch.zeros(n, T2.NF), rr.to(torch.float32)], -1).to(self.dev, x.dtype)
            fr = fr.view(n, 1, -1).expand(n, L, -1).reshape(n * L, -1)
            add = net.vhead(xf.reshape(n * L, -1), xt.reshape(n * L, -1), fr).float().view(n, L)
            base = (lg + add).masked_fill(~mask, float("-inf"))
            lp = torch.log_softmax(base, -1)
            ent = -(lp.exp() * lp.nan_to_num(neginf=0.0)).sum(-1)
        return base.clone(), xf.clone(), xt.clone(), x.mean(1).clone(), ent.clone()


def student(base_net, P, cad, ix, sq, m, anchors):
    """B's passes at the anchor ratings (one pass when anchors = (2000,)) -> base (B, L, A), xf, xt (B, L, 1024 A), xm (B, 1024 A),
    entropy of B's masked policy at the anchor nearest 2000 (B,)."""
    import torch
    if tuple(anchors) == (E_BODY,):
        b, xf, xt, xm, ent = base_net.run(P, cad, ix, sq, m)
        return b.unsqueeze(-1), xf, xt, xm, ent
    outs = [base_net.run(P, cad, ix, sq, m, E=torch.full((P.shape[0],), float(a))) for a in anchors]
    k0 = int(np.argmin([abs(a - E_BODY) for a in anchors]))
    return (torch.stack([o[0] for o in outs], -1), torch.cat([o[1] for o in outs], -1), torch.cat([o[2] for o in outs], -1),
            torch.cat([o[3] for o in outs], -1), outs[k0][4])


def pad(rows, L=None):
    """rows: list of 1-D int tensors -> (B, L) padded with 0 and a mask."""
    import torch
    L = L or max(len(r) for r in rows)
    out = torch.zeros(len(rows), L, dtype=torch.long)
    m = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
        m[i, :len(r)] = True
    return out, m


def ctx_vec(es_best, ply, cad, nlegal, ent):
    return [es_best, ply / 100.0] + [1.0 if cad == c else 0.0 for c in LH.CADS] + [math.log(nlegal), float(ent)]


def load_gen(path, device="cpu"):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("meta", {}).get("format") != FORMAT:
        raise RuntimeError(f"{path}: not a Toby 2 level-head file ({FORMAT})")
    m = make_gen(blob["meta"]["du"], blob["meta"]["hidden"], blob["meta"].get("arch", "bilinear"), len(blob["meta"].get("anchors", [E_BODY])))
    m.load_state_dict(blob["state"], strict=True)
    return m.eval().to(device), blob["meta"]


class GenService:
    """Engine side, one move at a time (Siby's per-move requests): B once at E = Eo = 2000 -> the move's curve over GRID (move term +
    expected-score term when values are given) and, from the same pass, the model's move distribution at any rating."""

    def __init__(self, net, head_path):
        import torch
        self.torch = torch
        self.net = net
        self.base = Base(net)
        self.model, self.meta = load_gen(head_path, net.device)

    def move(self, board, uci, cad, best=None, played=None, dist_elo=None):
        """board: the position BEFORE the move, with its move stack. -> dict(curve (73,), move (73,), loss (73,) or None,
        dist {uci: p} at dist_elo or None, p_played at dist_elo)."""
        torch = self.torch
        enc = self.net.encode(board)
        if uci not in enc["legal"]:
            raise ValueError(f"illegal move {uci}")
        ix, sq = self.base.indices(enc["planes"], enc["transform"], enc["mi"])
        m = torch.ones(1, len(ix), dtype=torch.bool)
        base, xf, xt, xm, ent = student(self.base, enc["planes"], [LH.CADS.index(cad)], ix.view(1, -1), sq.view(1, -1, 2), m,
                                        self.meta.get("anchors", [E_BODY]))
        dev = self.net.device
        rbf = self.base.rbf_grid.to(dev).unsqueeze(0)
        j = enc["legal"].index(uci)
        out = {}
        with torch.no_grad():
            lp = torch.log_softmax(self.model.move_logits(base, xf, xt, xm, rbf), -1)[0]      # (73, L)
            mv = lp[:, j].float().cpu().numpy()
            out["move"] = mv
            k, esb = loss_bin(best, played)
            if k >= 0:
                ctx = torch.tensor([ctx_vec(esb, board.ply(), cad, len(ix), float(ent[0]))], device=dev)
                ll = self.model.loss_logits(xm, ctx, rbf)[0, :, k].float().cpu().numpy()
                out["loss"] = ll
                out["curve"] = mv + ll
            else:
                out["loss"] = None
                out["curve"] = mv
            if dist_elo is not None:
                r = self.net.M.rating_rbf(torch.tensor([float(dist_elo)])).to(dev).view(1, 1, -1)
                p = torch.softmax(self.model.move_logits(base, xf, xt, xm, r)[0, 0], -1).float().cpu().numpy()
                out["dist"] = dict(zip(enc["legal"], p.tolist()))
                out["p_played"] = float(p[j])
        return out
