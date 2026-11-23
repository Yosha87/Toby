# -*- coding: utf-8 -*-
"""Toby 2 level head (Maia-free rating estimate): shared runtime code for data preparation, training, dev evaluation and the engine.

Per half-move, Toby 2's body B runs ONCE, at a fixed reference input that carries no rating information about the game:
mover E = opponent Eo = E_BODY (2000), the game's cadence, venue over the board, clocks / base / increment unknown (the engine's
analysis path), the game's history. B's ratings are inputs (two RBF vectors, model.rating_rbf), so they must be fixed: 2000 is the middle
of the 1200-3000 grid and of the training population.

From that one pass the head reads:
  * the body tokens of the played move's from / to squares (as B's per-move MLP reads them) and the mean of the 64 tokens (position
    context): 3 x 1024 numbers;
  * 14 network scalars: log p(played) of B's values-masked move head at the 10 reference ratings REF (the head only is re-run with its
    rating inputs at each REF; the body stays at E_BODY), the entropy and the top log-probability of the masked distribution at E_BODY,
    log(number of legal moves), log(1 + rank of the played move);
  * 4 context scalars: ply / 100, cadence one-hot (classical, rapid, blitz);
  * 6 Stockfish scalars, when available: a "values present" flag, the expected-score loss of the played move at R = 1500 / 2000 / 2500
    (CLAUDE.md formula: ES(R, best) - ES(R, played), mates +-10000 uncapped, other scores capped at +-2000; never raw centipawns), the
    expected score of the best move at 1500, and "played = best". Without values all six are 0 (the flag says "missing"); the head was
    trained with the values removed on a random share of rows, so it runs either way.
It outputs a per-move log-likelihood curve over GRID (1200-3000, step 25): log q(E | move) - log pi(E), q = the head's rating
distribution (softmax over GRID of a smooth RBF expansion, centres every 100, width 100), pi = the training rows' label distribution.
A player's curve is the SUM of its moves' curves (Bayesian, full weight); the calibration (prior, temperature, correction, E_R) is
analyse.py's family, outside this module."""
import math

import numpy as np

E_BODY = 2000.0
REF = tuple(float(e) for e in range(1200, 3001, 200))
GRID = np.arange(1200, 3001, 25, dtype=float)
CENTRES = np.arange(1200, 3001, 100, dtype=float)
WIDTH = 100.0
ES_R = (1500.0, 2000.0, 2500.0)
CADS = ("classical", "rapid", "blitz")
N_TOK = 3 * 1024
N_NET, N_CTX, N_SF = 14, 4, 6
N_SCAL = N_NET + N_CTX + N_SF
FORMAT = "toby2-levelhead/2"


# ------------------------------------------------------------------------------------------------ expected score (CLAUDE.md)
def score_to_cp(v):
    """Mover's-view value, mates +-10000 -> mates stay +-10000 (uncapped), other scores capped at +-2000 (find_maia_blunders.score_to_cp)."""
    v = float(v)
    if abs(v) >= 10000:
        return 10000.0 if v > 0 else -10000.0
    return max(-2000.0, min(2000.0, v))


def expected_score(rating, cp):
    """ES(R, E) in % for the side to move (find_maia_blunders.expected_score)."""
    c = rating * -0.00000274 + 0.00048
    x = c * cp
    if x > 700:
        return 0.0
    return 100 * (0.5 + 0.5 * (2 / (1 + math.exp(x)) - 1))


def sf_scalars(best_cp, played_cp):
    """The 6 Stockfish scalars from the best move's and the played move's values (mover's view, mates +-10000); None -> missing."""
    if best_cp is None or played_cp is None:
        return [0.0] * N_SF
    b, p = score_to_cp(best_cp), score_to_cp(played_cp)
    if p > b:   # a played move valued above the reported best (another search, or MultiPV noise): it is the best
        b = p
    loss = [(expected_score(r, b) - expected_score(r, p)) / 100.0 for r in ES_R]
    return [1.0] + loss + [expected_score(1500.0, b) / 100.0, 1.0 if p >= b else 0.0]


def sf_scalars_torch(best, played):
    """sf_scalars on tensors of raw values (mover's view, mates +-10000; NaN = missing) -> (n, 6). Same formula, for training-time noise."""
    import torch
    miss = torch.isnan(best) | torch.isnan(played)
    cap = lambda v: torch.where(v.abs() >= 10000, torch.sign(v) * 10000.0, v.clamp(-2000.0, 2000.0))
    b, p = cap(torch.nan_to_num(best)), cap(torch.nan_to_num(played))
    b = torch.maximum(b, p)
    es = lambda r, v: 100 * (0.5 + 0.5 * (2 / (1 + torch.exp((r * -0.00000274 + 0.00048) * v)) - 1))
    cols = [torch.ones_like(b)] + [(es(r, b) - es(r, p)) / 100.0 for r in ES_R] + [es(1500.0, b) / 100.0, (p >= b).float()]
    out = torch.stack(cols, 1)
    return torch.where(miss.view(-1, 1), torch.zeros_like(out), out)


def ctx_scalars(ply, cad):
    return [ply / 100.0] + [1.0 if cad == c else 0.0 for c in CADS]


def basis():
    return np.exp(-((GRID[:, None] - CENTRES[None, :]) / WIDTH) ** 2)   # (73, 19)


# ------------------------------------------------------------------------------------------------ the head
def make_head(hidden=256):
    import torch
    import torch.nn as nn

    class LevelHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Sequential(nn.Linear(N_TOK, hidden), nn.LayerNorm(hidden))
            self.sc = nn.Linear(N_SCAL, 64)
            self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden + 64, hidden), nn.GELU(), nn.Linear(hidden, len(CENTRES)))
            self.register_buffer("phi", torch.tensor(basis(), dtype=torch.float32))
            self.register_buffer("smu", torch.zeros(N_SCAL))
            self.register_buffer("ssd", torch.ones(N_SCAL))
            self.register_buffer("logpi", torch.zeros(len(GRID)))

        @staticmethod
        def transform(scal):
            """The 10 log p at REF vary little with REF (B's rating enters through the body): their mean and 100 x the deviations of
            the first 9 from it, then the other scalars unchanged."""
            scal = scal.float()
            r = scal[:, :len(REF)]
            m = r.mean(1, keepdim=True)
            return torch.cat([m, (r - m)[:, :len(REF) - 1] * 100.0, scal[:, len(REF):]], -1)

        def logq(self, tok, scal):
            """(B, 3072), (B, 24) raw scalars [net 14, ctx 4, sf 6] -> (B, 73) log q(E | move)."""
            s = (self.transform(scal) - self.smu) / self.ssd
            z = torch.cat([self.tok(tok.float()), self.sc(s)], -1)
            a = self.mlp(z)
            return torch.log_softmax(a @ self.phi.T, -1)

    class LevelPair(nn.Module):
        """Two heads of the same form trained on the same rows: the MOVE head q(E | position, move) and the POSITION head q(E | position)
        (the move's tokens and every move-specific scalar removed). The per-move curve is their log ratio,
            log q(E | position, move) - log q(E | position) = log p(move | position, E) + const(E-free),
        the likelihood of the move itself: the rating information of the position (shared by both players, made by both) cancels, and
        so does the training label distribution."""

        def __init__(self):
            super().__init__()
            self.mv, self.ps = LevelHead(), LevelHead()

        @staticmethod
        def pos_inputs(tok, scal):
            t = tok.clone()
            t[:, :2 * 1024] = 0
            s = scal.clone()
            s[:, POS_ZERO] = 0
            return t, s

        def logq(self, tok, scal):
            return self.mv.logq(tok, scal)

        def logq_pos(self, tok, scal):
            return self.ps.logq(*self.pos_inputs(tok, scal))

        def curve(self, tok, scal):
            """Per-move log-likelihood over GRID."""
            return self.logq(tok, scal) - self.logq_pos(tok, scal)

    return LevelPair()


# raw scalar columns removed for the position head: log p(played) at REF (0-9), rank of the played move (13), the losses (19-21) and
# "played = best" (23); kept: entropy / top log p / legal moves (10-12), ply and cadence (14-17), values flag (18), ES(1500, best) (22)
POS_ZERO = list(range(10)) + [13, 19, 20, 21, 23]


def save_head(head, path, meta):
    import torch
    torch.save({"meta": {"format": FORMAT, "E_body": E_BODY, "ref": REF, "grid": GRID.tolist(), "centres": CENTRES.tolist(),
                         "width": WIDTH, "es_r": ES_R, **meta},
                "state": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, path)


def load_head(path, device="cpu"):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("meta", {}).get("format") != FORMAT:
        raise RuntimeError(f"{path}: not a Toby 2 level-head file ({FORMAT})")
    h = make_head(int(blob["meta"].get("hidden", 256)))
    h.load_state_dict(blob["state"], strict=True)
    return h.eval().to(device), blob["meta"]


# ------------------------------------------------------------------------------------------------ features from B's one pass
class Extractor:
    """B's body once per position at E = Eo = E_BODY, batched over positions; the head's token and network inputs."""

    def __init__(self, net):
        """net: toby2.Toby2Net (any device / dtype)."""
        import torch
        import toby2 as T2
        self.torch, self.T2, self.tn = torch, T2, net
        self.M = net.M
        self.dev = net.device
        self.dt = torch.float16 if net.half else torch.float32
        rr0 = torch.cat([self.M.rating_rbf(torch.tensor([E_BODY])), self.M.rating_rbf(torch.tensor([E_BODY]))], -1)   # (1, 24)
        self.rr0 = rr0
        R = torch.tensor(REF)
        self.rref = torch.cat([self.M.rating_rbf(R), self.M.rating_rbf(R)], -1)                                     # (K, 24), Eo = E
        self.sides = {c: net.side_inputs(None, None, None, None, c, "otb") for c in CADS}

    def encode(self, board):
        return self.tn.encode(board)

    def run(self, encs, played, cads):
        """encs: list of Toby2Net.encode dicts, played: list of uci, cads: list of cadence -> (tok (B, 3072) float16 numpy,
        net scalars (B, 14) float32 numpy)."""
        torch, T2 = self.torch, self.T2
        net = self.tn.net
        n = len(encs)
        with torch.inference_mode():
            P = torch.cat([e["planes"] for e in encs], 0).to(self.dev, self.dt)
            side = torch.cat([self.sides[c] for c in cads], 0).to(self.rr0.dtype)
            extra = torch.cat([self.rr0.expand(n, -1), side], -1).to(self.dev, self.dt)
            x = net.lc0.body(P, extra, net.extra_lin, net.inject(extra) if net.freeze > 0 else None, net.freeze)   # (B, 64, 1024)
            grid = net.lc0.policy_grid(x)                                                                         # (B, 4288)
            K = len(REF)
            tok = np.zeros((n, N_TOK), np.float16)
            sc = np.zeros((n, N_NET), np.float32)
            fr_ref = torch.cat([torch.zeros(K, T2.NF), self.rref], -1)        # masked value features (all zero) + ratings at REF
            fr_0 = torch.cat([torch.zeros(1, T2.NF), self.rr0], -1)
            for b in range(n):
                e = encs[b]
                t = e["transform"]
                idx = T2.LT.castle_fix(net.vt[t:t + 1].cpu(), e["planes"])[0, e["mi"]].to(self.dev)
                sq = net.sqt[t].cpu()[e["mi"]].to(self.dev)
                L = len(e["legal"])
                j = e["legal"].index(played[b])
                xb = x[b]
                lg = grid[b, idx].float()                                                # (L,)
                fr = torch.cat([fr_0, fr_ref], 0).to(self.dev)                           # (1 + K, 31)
                fr = fr.view(1 + K, 1, -1).expand(1 + K, L, -1).reshape((1 + K) * L, -1)
                xf = xb[sq[:, 0]].repeat(1 + K, 1)
                xt = xb[sq[:, 1]].repeat(1 + K, 1)
                add = net.vhead(xf, xt, fr.to(xb.dtype)).float().view(1 + K, L)
                if net.vpath is not None:
                    add = add + net.vpath(fr.to(net.vpath[0].weight.dtype)).squeeze(-1).float().view(1 + K, L)
                lp = torch.log_softmax(lg.view(1, L) + add, -1)                          # (1 + K, L)
                l0 = lp[0]
                p0 = l0.exp()
                rank = int((l0 > l0[j]).sum())
                sc[b, :K] = lp[1:, j].cpu().numpy()
                sc[b, K] = float(-(p0 * l0).sum())
                sc[b, K + 1] = float(l0.max())
                sc[b, K + 2] = math.log(L)
                sc[b, K + 3] = math.log1p(rank)
                tok[b] = torch.cat([xb[sq[j, 0]], xb[sq[j, 1]], xb.mean(0)]).float().cpu().numpy().astype(np.float16)
        return tok, sc


# ------------------------------------------------------------------------------------------------ engine side
class LevelService:
    """One move at a time, as Siby's per-move requests: B once at E_BODY + the head -> the move's curve over GRID. Optional: B's
    values-masked move distribution at a given rating (one more body pass at E = Eo = that rating)."""

    def __init__(self, net, head_path):
        import torch
        self.torch = torch
        self.net = net
        self.X = Extractor(net)
        self.head, self.meta = load_head(head_path, "cpu")

    def move(self, board, uci, cad, best=None, played=None):
        """board: the position BEFORE the move, with its move stack -> (curve over GRID, numpy (73,); the encoding, for dist)."""
        torch = self.torch
        enc = self.X.encode(board)
        if uci not in enc["legal"]:
            raise ValueError(f"illegal move {uci}")
        tok, sc = self.X.run([enc], [uci], [cad])
        s = sc[0].tolist() + ctx_scalars(board.ply(), cad) + sf_scalars(best, played)
        with torch.no_grad():
            c = self.head.curve(torch.from_numpy(tok), torch.tensor([s], dtype=torch.float32))[0].numpy()
        return c, enc

    def dist(self, enc, cad, elo):
        """B's values-masked move distribution at mover = opponent = elo: {uci: p}."""
        side = self.net.side_inputs(None, None, None, None, cad, "otb")
        bo = self.net.body(enc, float(elo), float(elo), side)
        f, s = self.net.features_masked(enc, float(elo))
        lp, _ = self.net.heads(enc, bo, f, s)
        return dict(zip(enc["legal"], np.exp(lp).tolist()))


def _basis(r):
    r = min(max(float(r), 1400.0), 2800.0)
    return np.array([1.0, r] + [max(0.0, r - k) for k in (1800, 2200, 2500)])


def _post(L, mu, sd):
    if sd is not None:
        L = L - 0.5 * ((GRID - mu) / sd) ** 2
    L = L - L.max()
    w = np.exp(L)
    w /= w.sum()
    m = float(w @ GRID)
    return m, float(np.sqrt(max(w @ GRID ** 2 - m * m, 1.0)))


def estimate(S, group_cal, given=None):
    """A frozen group calibration (lh_freeze.py) applied to a player's summed curve S over GRID -> {E_A, sd_A[, E_R]}."""
    a = group_cal["E_A"]
    x, sx = _post(np.asarray(S, float) / a["tau"], a["prior_mean"], a["prior_sd"])
    if a["correction"] == "lin":
        e, slope = a["coef"][0] + a["coef"][1] * x, a["coef"][1]
    else:
        f = lambda v: float(_basis(v) @ np.array(a["coef"]))
        e, slope = f(x), (f(x + 1) - f(x - 1)) / 2
    out = {"E_A": e, "sd_A": abs(slope) * sx}
    if given is not None:
        r = group_cal["E_R"]
        m, _ = _post(np.asarray(S, float) / r["tau"], float(given), None if r["s"] == "flat" else r["s"])
        out["E_R"] = m - float(_basis(given) @ np.array(r["drift_coef"]))
    return out
