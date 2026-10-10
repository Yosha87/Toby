# -*- coding: utf-8 -*-
r"""
toby_uci.py - Toby, The Oracle By Yosha: UCI engine (see README.md and docs/METHOD.md).

Idea: Maia-3 gives the move distribution of a player of strength E; Leela (lc0) gives the value of each move; Toby's rule corrects Maia:
    score = log p - lambda * loss/100 + gamma * [Leela's best move] (+ a minimum prior for that move from E = 2200)      then softmax
  Classical: Maia at min(E+700, 2900); lambda = max(0, 3.44 + 5.25 (E-2000)/1000); gamma = 0.43
  Rapid:     Maia at min(2800, Lichess-rapid equivalent of E); lambda = max(0, (E-2100)/72.5); gamma = 0.43 min(1, lambda)
  Blitz:     Maia at min(2800, Lichess-blitz equivalent of E); lambda = max(0, (E-2200)/98);   gamma = 0.26 min(1, lambda)
  (validated on sealed game sets, see docs/METHOD.md; below its threshold Toby plays like Maia.)
E = FIDE-equivalent strength of the rating entered (ChessGoals tables, toby_rating_convert.py).

Display: 1) Maia alone (instant), 2) Leela's static value of the plausible moves, 3) Leela 1024 nodes, then 64-node complements for the moves it did not explore, then the final distribution
and Toby's win/draw/loss evaluation of each move (davidson model, six sets of constants, see wdl_model).
Ratings: the options White Rating / Black Rating (0 = not set). Both set: each side has its own rating; the side to move is the "player", the other one the opponent, whose rating Maia receives
(moves: policy; evaluation: value head and an explicit rating-difference term). Only one set: both sides get that rating (the opponent is assumed to have the player's rating, the validated setting).
None set: both sides get 2000.
UCI options: White Rating, Black Rating, Rating Source, Time Control, Display (ChessBase | Nibbler | Standard), MultiPV, UCI_ShowWDL.
Default paths can be changed with the environment variables TOBY_LC0, TOBY_NET, TOBY_MAIA_REPO, TOBY_DEVICE.
"""
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess

HERE = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toby_rating_convert import from_fide, to_fide  # noqa: E402

NAME = "Toby - The Oracle By Yosha"


def detect_gui():
    """Name of the interface that launched the engine (Nibbler, ChessBase), found by walking up the parent processes; None if unknown."""
    try:
        import ctypes
        from ctypes import wintypes

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]

        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        procs = {}
        pe = PE()
        pe.dwSize = ctypes.sizeof(PE)
        ok = k32.Process32First(snap, ctypes.byref(pe))
        while ok:
            procs[pe.th32ProcessID] = (pe.th32ParentProcessID, pe.szExeFile.decode("mbcs", "ignore").lower())
            ok = k32.Process32Next(snap, ctypes.byref(pe))
        k32.CloseHandle(snap)
        pid = os.getpid()
        for _ in range(8):
            if pid not in procs:
                break
            ppid, _name = procs[pid]
            pid = ppid
            name = procs.get(pid, (0, ""))[1]
            if "nibbler" in name:
                return "Nibbler"
            if "chessbase" in name or "cbase" in name or "playchess" in name or "fritz" in name:
                return "ChessBase"
    except Exception:  # noqa: BLE001
        pass
    return None


GUI_DETECTED = detect_gui()
DEFAULT_DISPLAY = os.environ.get("TOBY_DEFAULT_DISPLAY") or GUI_DETECTED or "Standard"  # unknown program: plain engine behaviour (evaluation scores)
if DEFAULT_DISPLAY == "Nibbler":
    NAME += " (Lc0-compatible)"  # Nibbler recognises a Leela by the engine name: it then forces MultiPV = 500 and reads the per-move statistics
LC0 = os.environ.get("TOBY_LC0") or next((str(c) for c in (HERE / "lc0" / "lc0.exe", HERE / "lc0.exe") if c.exists()), "lc0")   # next to Toby, else on the PATH, or TOBY_LC0
NET = os.environ.get("TOBY_NET") or next((str(c) for c in sorted((HERE / "nets").glob("*.pb.gz")) + sorted(HERE.glob("*.pb.gz"))), str(HERE / "nets" / "BT3-768x15x24h-swa-2790000.pb.gz"))
MAIA_REPO = os.environ.get("TOBY_MAIA_REPO", str(HERE / "maia3"))   # a clone of https://github.com/CSSLab/maia3
DEVICE = os.environ.get("TOBY_DEVICE", "cpu")
MIN_N = 8  # a move needs this many Leela visits for its value to be trusted without a complementary search
LOG = os.environ.get("TOBY_LOG")

SOURCES = ["FIDE", "Lichess Classical", "Lichess Rapid", "Lichess Blitz", "Chess.com Rapid", "Chess.com Blitz"]
TIME_CONTROLS = ["Classical", "Rapid", "Blitz"]
DISPLAYS = ["ChessBase", "Nibbler", "Standard"]
DEFAULT_RATING = 2000   # used when neither White Rating nor Black Rating is set
opts = {"white_rating": 0, "black_rating": 0, "source": "FIDE", "mode": "Classical", "display": DEFAULT_DISPLAY, "move_choice": "prediction",
        "MultiPV": 500, "UCI_ShowWDL": True}

REAL_OUT = sys.stdout  # the UCI stream; stray prints from libraries (Maia) go to stderr
sys.stdout = sys.stderr
lock = threading.Lock()
stop_flag = threading.Event()
board = chess.Board()
worker = None
PRIOR = {}  # Maia's probabilities before Toby's correction (shown as P in Nibbler)
EV_LAST = {}  # last evaluation of each move in the current search (see emit)
LISTED = set()  # moves already listed to Nibbler in the current search
maia = None
lc0 = None
PREFETCH_ABORT = threading.Event()
ready = threading.Event()
load_error = None


def _stamp():
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(1000 * (t % 1)):03d}"


def out(s):
    with lock:
        REAL_OUT.write(s + "\n")
        REAL_OUT.flush()
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} < " + s + "\n")


# ----------------------------------------------------------------------------------------------------- player strength
def strength_E(rating, source):
    """FIDE-equivalent strength E of the rating entered."""
    if source == "FIDE":
        return float(rating)
    platform, cadence = source.lower().split()
    platform = "chesscom" if platform.startswith("chess.com") else "lichess"
    return float(to_fide(rating, platform, cadence))


def rule_params(E):
    """(Maia input rating, lambda, gamma) for the current time control (rule form, lambda, gamma and the Maia input: validated on sealed sets, Classical, Rapid v1 and Blitz v1, 2026-10-06)."""
    mode = opts["mode"]
    if mode == "Classical":
        z = (E - 2000) / 1000
        return int(round(min(E + 700, 2900))), max(0.0, 3.44 + 5.25 * z), 0.43
    if mode == "Rapid":  # lambda = (E-2100)/72.5; gamma = 0.43 min(1, lambda); Maia at the Lichess-rapid equivalent, capped at 2800
        x = int(round(min(2800.0, from_fide(E, "lichess", "rapid"))))
        lam = max(0.0, (E - 2100) / 72.5)
        return x, lam, 0.43 * min(1.0, lam)
    x = int(round(min(2800.0, from_fide(E, "lichess", "blitz"))))  # Blitz: lambda = (E-2200)/98; gamma = 0.26 min(1, lambda)
    lam = max(0.0, (E - 2200) / 98)
    return x, lam, 0.26 * min(1.0, lam)


# ----------------------------------------------------------------------------------------------------- Maia
def preload_lc0():
    global lc0
    try:
        lc0 = Lc0()
    except Exception:  # noqa: BLE001
        lc0 = None


def prefetch_maia(b, dist):
    """Idle time: Maia's policy for the positions that can follow the likely moves, so that stepping forward shows Maia's line at once."""
    for u, p in sorted(dist.items(), key=lambda kv: -kv[1])[:8]:
        if p < 0.01 or PREFETCH_ABORT.is_set() or maia is None:
            return
        try:
            c = b.copy()
            c.push_uci(u)
            if c.is_game_over():
                continue
            E_c, opp_c = side_ratings(c)
            maia_policy(c, rule_params(E_c)[0], opp_c[1] if opp_c else None)
        except Exception:  # noqa: BLE001
            return


def load_maia():
    global maia, load_error
    try:
        if MAIA_REPO not in sys.path:
            sys.path.insert(0, MAIA_REPO)
        import torch
        from maia3.dataset import get_legal_moves_mask
        from maia3.uci import Maia3UCIEngine, parse_args
        margs = ["--model", "maia3-79m", "--device", DEVICE, "--no-use-amp"]
        mw = os.environ.get("TOBY_MAIA_WEIGHTS") or next((str(c) for c in (HERE / "maia" / "maia3-79m.pt",) if c.exists()), None)   # local weights: no download
        if mw:
            margs += ["--checkpoint-path", mw]
        cfg = parse_args(margs)
        eng = Maia3UCIEngine(cfg)
        eng.ensure_model_loaded()
        maia = (eng, torch, get_legal_moves_mask)
    except Exception as exc:  # noqa: BLE001
        load_error = f"Maia: {exc}"
    ready.set()


MAIA_LOCK = threading.RLock()   # the Maia model keeps its board / history inside: one call at a time (search thread and idle-time prefetch)
_MP = {}                          # Maia policy by (position, rating, opponent rating): going back to a position, or stepping to a prefetched one, costs nothing


def maia_policy(b, elo, oppo=None):
    """Maia's move probabilities {uci: p} for a player of the given rating; oppo = the opponent's rating given to Maia (default: the same rating on both sides)."""
    k = (b.fen(), elo, oppo)
    r = _MP.get(k)
    if r is None:
        with MAIA_LOCK:
            r = _maia_policy_raw(b, elo, oppo)
        if len(_MP) > 3000:
            _MP.clear()
        _MP[k] = r
    return dict(r)


def _maia_policy_raw(b, elo, oppo=None):
    eng, torch, mask_fn = maia
    cfg = eng.cfg
    with torch.no_grad():
        eng.board = b
        eng._reset_history()
        tokens = eng._tokens_from_history(eng.history).unsqueeze(0).to(cfg.device)
        se = torch.tensor([elo], dtype=torch.long, device=cfg.device)
        so = se if oppo is None else torch.tensor([oppo], dtype=torch.long, device=cfg.device)
        logits_move, _v, _ = eng.model(tokens, se, so)
        logits = logits_move[0].float()
        mask = mask_fn(b, eng.all_moves_dict).to(cfg.device)
        probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        res = {}
        for idx in torch.nonzero(mask).flatten().tolist():
            mv = eng._move_from_index(idx)
            if mv is not None:
                res[mv.uci()] = float(probs[idx])
    return res


def maia_value(b, elo, oppo=None):
    """[loss, draw, win] estimated by Maia for the side TO MOVE in position b; oppo = the opponent's rating (default: the same rating on both sides)."""
    with MAIA_LOCK:
        return _maia_value_raw(b, elo, oppo)


def _maia_value_raw(b, elo, oppo=None):
    eng, torch, _ = maia
    cfg = eng.cfg
    with torch.no_grad():
        eng.board = b
        eng._reset_history()
        tokens = eng._tokens_from_history(eng.history).unsqueeze(0).to(cfg.device)
        se = torch.tensor([elo], dtype=torch.long, device=cfg.device)
        so = se if oppo is None else torch.tensor([oppo], dtype=torch.long, device=cfg.device)
        _, v, _ = eng.model(tokens, se, so)
        return torch.softmax(v[0].float(), dim=-1).tolist()


# ----------------------------------------------------------------------------------------------------- Leela
MOVE_RE = re.compile(r"info string (\S+)\s+\(\s*\d+\s*\)\s+N:\s*(\d+).*?\(P:\s*([\d.]+)%\).*?\(WL:\s*([-\d.]+)\).*?\(D:\s*([-\d.]+)\).*?\(Q:\s*([-\d.]+)\)")
ROOT_RE = re.compile(r"info string node.*?\(WL:\s*([-\d.]+)\).*?\(D:\s*([-\d.]+)\)")


class Lc0:
    """Thin wrapper around an lc0 process (UCI, --verbose-move-stats)."""

    def __init__(self):
        self.p = subprocess.Popen([LC0, f"--weights={NET}", "--backend=" + os.environ.get("TOBY_BACKEND", "cuda"), "--verbose-move-stats"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        self.new_game()

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        for line in self.p.stdout:
            if line.startswith(token):
                return

    def new_game(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait("readyok")

    def search(self, fen, nodes, searchmoves=None):
        """Returns ({uci: (visits, WL, D, Q)}, (WL, D) of the root). The tree is reset at each call (lc0 would otherwise reuse the tree of the same position)."""
        self.new_game()
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}" + (f" searchmoves {searchmoves}" if searchmoves else ""))
        moves, root = {}, None
        finished = threading.Event()

        def watch():  # the GUI's stop reaches lc0 at once: the partial result is discarded by the caller, which checks the flag
            while not finished.is_set():
                if stop_flag.wait(0.02):
                    if not finished.is_set():
                        try:
                            self.send("stop")
                        except Exception:  # noqa: BLE001
                            pass
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        for line in self.p.stdout:
            if line.startswith("bestmove"):
                break
            r = ROOT_RE.match(line)
            if r:
                try:
                    root = (float(r.group(1)), float(r.group(2)))
                except ValueError:
                    pass
                continue
            m = MOVE_RE.match(line)
            if m:
                try:
                    moves[m.group(1)] = (int(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
                except ValueError:  # unvisited move: values are '-.-----'
                    pass
        finished.set()
        watcher.join()
        # lc0 prints a castling move as "king takes own rook" (e1h1, e1a1, e8h8, e8a8); Maia and UCI use e1g1, e1c1, e8g8, e8c8
        try:
            b0 = chess.Board(fen)
            for u in list(moves):
                f, t = chess.parse_square(u[:2]), chess.parse_square(u[2:4])
                if len(u) == 4 and b0.piece_type_at(f) == chess.KING and b0.piece_type_at(t) == chess.ROOK and b0.color_at(f) == b0.color_at(t):
                    moves[chess.square_name(f) + chess.square_name(chess.square(6 if chess.square_file(t) > chess.square_file(f) else 2, chess.square_rank(f)))] = moves.pop(u)
        except Exception:  # noqa: BLE001
            pass
        return moves, root

    def static_values(self, b, moves):
        """Value of each move from the mover's point of view without any search: the network's value of the position after the move (1 node, about 0.03 s per move).
        {uci: Q in [-1, 1]}. Used only for the quick provisional display, never for the final rule."""
        res = {}
        for u in moves:
            if stop_flag.is_set():
                break
            c = b.copy()
            c.push(chess.Move.from_uci(u))
            if c.is_checkmate():
                res[u] = 1.0
            elif c.is_game_over():
                res[u] = 0.0
            else:
                _, r = self.search(c.fen(), 1)
                if r:
                    res[u] = -r[0]  # r = (WL, D) for the side to move after the move, i.e. the opponent
        return res

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.p.kill()


# ----------------------------------------------------------------------------------------------------- move correction rule
RAMP_START = 2200.0   # one start for the three modes: best or equal in each of them on the tuning sets (Classical +0.0083, Rapid +0.0059, Blitz +0.0018 in log-likelihood), no strength band made worse


def ramp_factor(pm, best, E):
    """Leela's best move gets a minimum Maia prior that rises continuously with the strength E (tuning sets, 2026-10-07; to be confirmed on a sealed set).
    p1(E) = 0.08 (E - 2200) / 500, capped at 20%, zero below 2200; when the Maia prior p of the best move is below p1 it is replaced by p' = 0.8 p1 + 0.2 p
    (p'(0) = 0.8 p1, p'(p1) = p1, strictly increasing, no jump). The rule is multiplicative: the weight of that move is scaled by p' / p."""
    if best not in pm:
        return 1.0
    p1 = min(0.20, max(0.0, 0.08 * (E - RAMP_START) / 500))
    p = pm[best]
    if p1 <= 0 or p >= p1:
        return 1.0
    return (0.8 * p1 + 0.2 * p) / max(p, 1e-6)


def toby_distribution(pm, lee, extra, E):
    """pm: Maia probabilities {uci: p}; lee: {uci: (N, WL, D, Q)}; extra: {uci: Q} (64-node values); returns ({uci: toby probability}, {uci: expected-score loss}, best move).
    A move that has no Leela value at all keeps exactly its Maia probability (decision 2026-10-07); the moves that have one share the remaining probability according to the rule."""
    x, lam, gam = rule_params(E)
    best = max(lee, key=lambda u: (lee[u][0], lee[u][3])) if lee else None

    def q(u):
        v = lee.get(u)
        if v and v[0] >= MIN_N:
            return v[3]
        if u in extra:
            return extra[u]
        if v and v[0] >= 1:
            return v[3]
        return None

    qb = q(best) if best else None
    us = list(pm)
    loss = {u: (max(0.0, 50 * (qb - q(u))) if (qb is not None and q(u) is not None) else 0.0) for u in us}
    valued = [u for u in us if qb is not None and q(u) is not None] if qb is not None else list(us)
    unvalued = [u for u in us if u not in set(valued)]
    s = {u: math.log(max(pm[u], 1e-12)) - lam * loss[u] / 100 for u in valued}
    if best in s:
        s[best] += gam + math.log(ramp_factor(pm, best, E))
    m = max(s.values())
    e = {u: math.exp(v - m) for u, v in s.items()}
    z = sum(e.values())
    mass_unvalued = sum(pm[u] for u in unvalued)
    d = {u: pm[u] for u in unvalued}
    d.update({u: (1.0 - mass_unvalued) * e[u] / z for u in valued})
    return d, loss, best


def toby_distribution_static(pm, qs, E):
    """Quick provisional version of the rule: qs = {uci: static value Q from the mover's point of view} for the plausible moves; the best of them is the reference.
    Same formula as toby_distribution; a move without a static value keeps its Maia probability."""
    x, lam, gam = rule_params(E)
    best = max(qs, key=qs.get) if qs else None
    if best is None:
        return dict(pm)
    valued = [u for u in pm if u in qs]
    s = {u: math.log(max(pm[u], 1e-12)) - lam * max(0.0, 50 * (qs[best] - qs[u])) / 100 for u in valued}
    s[best] += gam + math.log(ramp_factor(pm, best, E))
    m = max(s.values())
    e = {u: math.exp(v - m) for u, v in s.items()}
    z = sum(e.values())
    unvalued = [u for u in pm if u not in qs]
    mass = sum(pm[u] for u in unvalued)
    d = {u: pm[u] for u in unvalued}
    d.update({u: (1.0 - mass) * e[u] / z for u in valued})
    return d


# ----------------------------------------------------------------------------------------------------- Toby's evaluation
def _logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


# Win / draw / loss model (Davidson), decided 2026-10-08 (tuned on open data, 5-fold CV by game; not yet sealed-tested):
#   t = (wL + wU max(0, (E - 2200)/1000)) eL + wM eM + kE eL |eL| + wD (E - E_opp)/1000        eL, eM = logit of Leela's / Maia's expected score for the side to move
#   P(win) = e^(t/2)/Z, P(loss) = e^(-t/2)/Z, P(draw) = nu/Z, ln nu = a2 + b2 |eL| + d2 logit(Maia draw). No starting advantage (equal inputs -> 50%).
# Constants per time control AND venue (decided 2026-10-08 after the final sealed test): the draw rate of the same strength differs between over-the-board and online games
# (balanced positions: Rapid 38-50% draws over the board vs 17-24% online; Blitz 24% vs 15%). Classical: one set (over the board only: no online data). Rapid: separate constants per venue;
# Blitz: shared constants, online games get a lower draw level (a2). The venue follows the Rating Source option: FIDE = over the board, Lichess / Chess.com = online.
# For each: one set for SINGLE-rating use (opponent not given: Maia gets the same rating on both sides, wD = 0) and one for use with the OPPONENT's rating
# (Maia's value head gets it, plus the explicit wD term). Maia's value head never gets a rating above 2700.
# Fitted on all development data (earlier fit data + the sealed games of 2026-10-08, which are no longer a test set): fit_venue_constants.py. NOT yet sealed-tested.
#                (wL,     wM,     a2,     b2,      d2,     wU,     kE,      wD)
WDL_SINGLE = {"Classical": {"otb": (1.1087, 0.4522, 2.3816, -0.3891, 0.7813, 0.8243, -0.1074, 0.0)},
              "Rapid":     {"otb": (0.8569, 0.5477, 1.6984, -0.2431, 0.7914, 0.8849, -0.0810, 0.0),
                            "online": (0.4445, 0.4395, 1.1045, -0.0651, 0.9957, 0.6563, -0.0273, 0.0)},
              "Blitz":     {"otb": (0.3381, 0.3304, 0.8656, -0.0536, 0.9051, 0.4955, -0.0115, 0.0),
                            "online": (0.3381, 0.3304, 0.3024, -0.0536, 0.9051, 0.4955, -0.0115, 0.0)}}
WDL_OPPONENT = {"Classical": {"otb": (0.9972, 0.3565, 2.5221, -0.3869, 0.7462, 0.9382, -0.0851, 5.4076)},
                "Rapid":     {"otb": (0.6704, 0.3246, 1.8220, -0.2449, 0.7356, 0.9555, -0.0275, 4.9631),
                              "online": (0.4010, 0.4123, 1.1977, -0.0731, 0.9956, 0.4886, -0.0165, 2.5558)},
                "Blitz":     {"otb": (0.3565, 0.0338, 1.0003, -0.0583, 0.8775, 0.5046, 0.0032, 4.3923),
                              "online": (0.3565, 0.0338, 0.2989, -0.0583, 0.8775, 0.5046, 0.0032, 4.3923)}}
VALUE_ELO_CAP = 2700
MAX_ELO_GAP = 500   # largest Elo gap between the two players taken into account


def venue():
    """'online' when the rating source is a platform (Lichess / Chess.com) in Rapid or Blitz, else 'otb' (over the board; also every Classical game: no online calibration)."""
    return "online" if opts["source"] != "FIDE" and opts["mode"] != "Classical" else "otb"


def wdl_model(es_leela, maia_ldw, E, D=None):
    """(win, draw, loss) of the side TO MOVE in a position: es_leela = expected score Leela gives that side; maia_ldw = Maia's [loss, draw, win] for that side;
    E = strength of that side; D = its strength minus its opponent's, in Elo (None: opponent not given -> the single-rating constants)."""
    wl, wm, a2, b2, d2, wu, ke, wd = (WDL_SINGLE if D is None else WDL_OPPONENT)[opts["mode"]][venue()]
    l, d, w = maia_ldw
    eL, eM, md = _logit(es_leela), _logit(w + d / 2), _logit(d)
    t = (wl + wu * max(0.0, (E - 2200) / 1000)) * eL + wm * eM + ke * eL * abs(eL) + (wd * D / 1000 if D is not None else 0.0)
    nu = math.exp(a2 + b2 * abs(eL) + d2 * md)
    wv, lv = math.exp(t / 2), math.exp(-t / 2)
    z = wv + lv + nu
    return (wv / z, nu / z, lv / z)


def eval_elo(E, x):
    """Rating given to Maia's value head: E clamped to [1000, 2700] in Classical, the Maia input rating (at most 2700) otherwise."""
    return int(max(1000, min(VALUE_ELO_CAP, round(E)))) if opts["mode"] == "Classical" else int(min(VALUE_ELO_CAP, x))


_MV = {}


def maia_value_cached(board, elo, oppo=None):
    k = (board.fen(), elo, oppo)
    if k not in _MV:
        if len(_MV) > 4000:
            _MV.clear()
        _MV[k] = maia_value(board, elo, oppo)
    return _MV[k]


def toby_evals(b, E, x, moves, lee, extra, root, deadline, opp=None):
    """Toby's evaluation of each move, from the point of view of the player who plays it: {uci: (WL, D)} with WL = win - loss, plus that of the position itself.
    The position after the move is evaluated for the side that must play there (the opponent), where the model is calibrated, then flipped.
    opp = (E_opp, x_opp): the opponent's strength and Maia input rating when the user gave it (None: same rating on both sides)."""
    ev_elo = eval_elo(E, x)
    ev_opp = eval_elo(opp[0], opp[1]) if opp else None
    root_wl = root[0] if root else 0.0

    def q(u):
        v = lee.get(u)
        if v and v[0] >= MIN_N:
            return v[3]
        if u in extra:
            return extra[u]
        if v and v[0] >= 1:
            return v[3]
        return None

    res = {}
    for u in moves:
        if time.time() > deadline or stop_flag.is_set():
            break
        c = b.copy()
        c.push(chess.Move.from_uci(u))
        if c.is_checkmate():
            res[u] = (1.0, 0.0)
            continue
        if c.is_game_over():
            res[u] = (0.0, 1.0)
            continue
        qq = q(u)
        es_opp = (1 - (qq if qq is not None else root_wl)) / 2  # Leela's value for the opponent, who is to move after the move
        if opp:
            wo, do, lo = wdl_model(es_opp, maia_value_cached(c, ev_opp, ev_elo), opp[0], opp[0] - E)
        else:
            wo, do, lo = wdl_model(es_opp, maia_value_cached(c, ev_elo), E)
        res[u] = (lo - wo, do)  # from the point of view of the player who played the move
    if opp:
        w, dr, l = wdl_model((1 + root_wl) / 2, maia_value_cached(b, ev_elo, ev_opp), E, E - opp[0])
    else:
        w, dr, l = wdl_model((1 + root_wl) / 2, maia_value_cached(b, ev_elo), E)
    return res, (w - l, dr)


# ----------------------------------------------------------------------------------------------------- display
def q_to_cp(q):
    return int(round(111.714640912 * math.tan(1.5620688421 * max(-0.99, min(0.99, q)))))


def move_value(u, lee, extra, root):
    """(WL, D) of a move from Leela, from the point of view of the side to move; None if unknown."""
    v = lee.get(u)
    if v and v[0] >= MIN_N:
        return v[1], v[2]
    if v and v[0] >= 1:
        return v[1], v[2]
    return None


def emit(stage, dist, lee, extra, root, nodes, t0, provisional, evals=None, root_eval=None, bd=None):
    fmt = opts["display"]
    if evals:
        EV_LAST.update(evals)   # a move keeps its last evaluation when a stage does not re-evaluate it: every listed move then has its WDL line in the same block (Nibbler sorts the moves that were in the latest block first)
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    keep = [(u, p) for u, p in items if p >= 0.001 or (fmt == "Nibbler" and u in LISTED)] or items[:1]
    if fmt == "Nibbler":   # a move listed at an earlier stage stays listed (with its new, smaller number): Nibbler keeps the old row, greyed, under the latest block otherwise
        LISTED.update(u for u, _ in keep)
    items = keep if fmt == "Nibbler" else keep[:int(opts["MultiPV"])]  # Nibbler sends its own MultiPV (3): ignored, every move above 0.1% is shown, like Leela
    ms = max(1, int((time.time() - t0) * 1000))
    # the depth is only a progress marker (Maia alone, Leela's static values, full rule). ChessBase draws no line below depth 10 (tested 2026-10-09: depth 6 lines are announced in its
    # header but never drawn, whatever wdl / nodes / time they carry), so its stages start at 10
    depth = {1: 10, 2: 13, 3: 16}[stage] if fmt == "ChessBase" else {1: 6, 2: 10, 3: 16}[stage]
    note = " (provisional)" if provisional else ""
    if fmt == "Nibbler":
        for k, (u, p) in reversed(list(enumerate(items))):  # like lc0: least visited move first (Nibbler reads the list backwards)
            val = EV_LAST.get(u)   # only Toby's evaluation is ever displayed, never Leela's raw value
            wl, d = val if val else (0.0, 0.0)   # no value of its own: neutral numbers, never the position's value
            q = extra.get(u, wl) if not val else wl
            out(f"info string {u:<5s} ({k + 1:<4d}) N: {int(round(p * 100000)):7d} (+ 0) (P: {100 * PRIOR.get(u, p):5.2f}%) (WL: {wl:8.5f}) (D: {d:5.3f}) (M:  60.0) (Q: {q:8.5f}) "
                f"(U: 0.01000) (S: {q + 0.01:8.5f}) (V: {q:7.4f})")
        wl, d = root_eval if root_eval else (root if root else (0.0, 0.0))
        out(f"info string node  (  1) N: {100000:7d} (+ 0) (P:  0.00%) (WL: {wl:8.5f}) (D: {d:5.3f}) (M:  60.0) (Q: {wl:8.5f}) (V: {wl:7.4f})")
        # classic MultiPV lines with wdl, one per move: Nibbler reads each move's WDL there (the "info string" lines do not carry it)
        for k, (u, p) in enumerate(items):
            val = EV_LAST.get(u)   # only Toby's evaluation is ever displayed, never Leela's raw value
            wl, d = val if val else (None, None)   # no WDL line for a move without a value: the interface shows '?'
            if wl is None:
                continue
            w, l = (1 - d + wl) / 2, (1 - d - wl) / 2
            out(f"info depth {depth} seldepth {depth} multipv {k + 1} score cp {q_to_cp(wl)} wdl {int(round(1000 * w))} {int(round(1000 * d))} {int(round(1000 * l))} "
                f"nodes 100000 nps {int(100000 / max(ms / 1000, 0.001))} time {ms} pv {u}")
    else:
        rows = []
        for u, p in items:
            val = EV_LAST.get(u)   # only Toby's evaluation is ever displayed, never Leela's raw value
            wl, d = val if val else (None, None)
            rows.append((u, p, wl, d))
        if fmt == "Standard":  # score = evaluation of the move, best first; no probabilities
            rows.sort(key=lambda r: -(r[2] if r[2] is not None else -9))
        for k, (u, p, wl, d) in enumerate(rows):
            cp = int(round(100 * p)) if (fmt == "ChessBase" or wl is None) else q_to_cp(wl)  # ChessBase: the score column carries the move probability (%)
            wdl = ""
            if wl is not None and opts["UCI_ShowWDL"]:
                w, l = (1 - d + wl) / 2, (1 - d - wl) / 2
                wdl = f" wdl {int(round(1000 * w))} {int(round(1000 * d))} {int(round(1000 * l))}"
            out(f"info depth {depth} seldepth {depth} multipv {k + 1} score cp {cp}{wdl} nodes {nodes} nps {int(nodes / max(ms / 1000, 0.001))} time {ms} pv {u}")
    bd = bd if bd is not None else board
    txt = " ".join(f"{bd.san(chess.Move.from_uci(u))} {100 * p:.1f}%" for u, p in items[:6])
    out(f"info string Toby{note}: {txt}")


# ----------------------------------------------------------------------------------------------------- search
def pick_move(dist):
    """Most likely move by default; random draw from the probabilities only in the (hidden) human mode, meant for a future bot build."""
    if opts["move_choice"] == "human":
        us = list(dist)
        return random.choices(us, weights=[dist[u] for u in us])[0]
    return max(dist, key=dist.get)


def side_ratings(b):
    """(E of the side to move, (E, Maia input) of its opponent or None if the user gave no opponent rating). 'White Rating' / 'Black Rating' (0 = not set): both set = the opponent is given;
    only one set = both sides get that rating (opponent not given, assumed equal); none set = both get 2000."""
    wr, br = opts["white_rating"], opts["black_rating"]
    given = wr > 0 and br > 0
    if given:
        rw, rb = wr, br
    else:
        rw = rb = wr or br or DEFAULT_RATING
    mine, theirs = (rw, rb) if b.turn == chess.WHITE else (rb, rw)
    E = strength_E(mine, opts["source"])
    if not given:
        return E, None
    Eo = strength_E(theirs, opts["source"])
    Eo = min(max(Eo, E - MAX_ELO_GAP), E + MAX_ELO_GAP)   # the fits only cover gaps up to about 500 points (99th percentile 550-750): no extrapolation beyond
    return E, (Eo, rule_params(Eo)[0])


def search(movetime, infinite=False):
    global lc0
    t0 = time.time()
    b = board.copy()
    legal = [m.uci() for m in b.legal_moves]
    if not legal:
        out("bestmove 0000")
        return
    if not ready.is_set():
        ready.wait()
    if maia is None:
        out(f"info string ERROR {load_error}")
        out(f"bestmove {legal[0]}")
        return
    E, opp = side_ratings(b)
    x, lam, gam = rule_params(E)
    emit_ = lambda *a, **k: emit(*a, bd=b, **k)   # the board of THIS search (the global one may already hold the next position)
    opp_txt = f"; opponent strength E = {opp[0]:.0f} (Maia input {opp[1]})" if opp else "; opponent: same rating"
    out(f"info string Toby: rating {E:.0f} ({opts['source']}) -> strength E = {E:.0f}; time control {opts['mode']}; "
        f"Maia input rating {x}; severity lambda = {lam:.2f}, bonus gamma = {gam:.2f}{opp_txt}")
    pm = maia_policy(b, x, opp[1] if opp else None)
    PRIOR.clear()
    PRIOR.update(pm)
    EV_LAST.clear()
    LISTED.clear()
    dist = dict(pm)
    emit_(1, dist, {}, {}, None, 1, t0, True)  # step 1: Maia alone (nodes 1: a GUI may ignore a line with nodes 0)
    best_dist = dist
    if len(legal) > 1 and not stop_flag.is_set():
        try:
            if lc0 is None:
                lc0 = Lc0()
            fen = b.fen()
            # step 2 (about 0.5 s): quick provisional answer from the network's static value of every plausible move; every plausible move has a value, none escapes the penalty
            plausible = [u for u, p in sorted(pm.items(), key=lambda kv: -kv[1]) if p >= 0.005]
            qs = lc0.static_values(b, plausible)
            sv = dict(qs)   # static values known so far

            def interim(dist, lee_, extra_, root_, with_root=True):
                """Toby's evaluation of the 12 most probable moves with the Leela values available at this stage (static, then search, then complements). Maia's values are cached."""
                sh = [u for u, p in sorted(dist.items(), key=lambda kv: -kv[1]) if p >= 0.001][:12]
                missing = [u for u in sh if u not in sv and not (lee_.get(u, (0,))[0] >= 1 or u in extra_)]
                if missing:
                    sv.update(lc0.static_values(b, missing))
                ev_, rev_ = toby_evals(b, E, x, sh, lee_, {**sv, **extra_}, root_, time.time() + 12, opp)
                return ev_, (rev_ if with_root else None)
            if qs and not stop_flag.is_set():
                d2 = toby_distribution_static(pm, qs, E)
                ev2, _ = interim(d2, {}, {}, None, with_root=False)
                emit_(2, d2, {u: (1, q, 0.0, q) for u, q in qs.items()}, {}, None, len(qs), t0, True, ev2, None)
                best_dist = d2
            if not stop_flag.is_set():
                lee, root = lc0.search(fen, 1024)  # step 3: the main search
                if stop_flag.is_set():  # stopped while it ran: keep the last complete answer (never a distribution built from missing values)
                    out(f"bestmove {pick_move(best_dist)}")
                    return
                need = [u for u in plausible if lee.get(u, (0,))[0] < MIN_N]
                extra = {}  # complementary 64-node values: the validated rule, and the only ones used for the final answer
                # provisional: moves not yet complemented keep their static value, so none is favoured while waiting
                d3, _, _ = toby_distribution(pm, lee, {u: qs[u] for u in need if u in qs}, E)
                ev3, rev3 = interim(d3, lee, {u: qs[u] for u in need if u in qs}, root)
                emit_(3, d3, lee, {u: qs[u] for u in need if u in qs}, root, 1024, t0, True, ev3, rev3)
                best_dist = d3
                for k, u in enumerate(need, 1):
                    if stop_flag.is_set():
                        break
                    out(f"info string Toby: checking the moves Leela did not explore, {k}/{len(need)}")
                    mv, _ = lc0.search(fen, 64, searchmoves=u)  # no time limit: the rule needs every complement
                    if stop_flag.is_set():  # interrupted complement: its partial value is not used
                        break
                    if u in mv and mv[u][0] >= 1:
                        extra[u] = mv[u][3]
                    prov = {**{v: qs[v] for v in need if v in qs}, **extra}
                    d3, _, _ = toby_distribution(pm, lee, prov, E)
                    best_dist = d3
                    if k < len(need):
                        ev3, rev3 = interim(d3, lee, prov, root)
                        emit_(3, d3, lee, prov, root, 1024, t0, True, ev3, rev3)
                if stop_flag.is_set():
                    out(f"bestmove {pick_move(best_dist)}")
                    return
                d3, _, _ = toby_distribution(pm, lee, extra, E)  # final: exactly the validated rule
                ev3, rev3 = interim(d3, lee, extra, root)
                emit_(3, d3, lee, extra, root, 1024, t0, True, ev3, rev3)  # full rule shown first, with the evaluation of the main moves; the complete evaluation follows
                best_dist = d3
                shown = [u for u, p in sorted(d3.items(), key=lambda kv: -kv[1]) if p >= 0.001][:30]
                # a shown move that Leela never valued gets its static value (1 node, about 0.03 s) for the evaluation; never the value of the position itself
                unvalued = [u for u in shown if not ((lee.get(u, (0,))[0] >= 1) or (u in extra))]
                stat = lc0.static_values(b, unvalued) if unvalued else {}
                evals, root_eval = toby_evals(b, E, x, shown, lee, {**stat, **extra}, root, time.time() + 12, opp)
                emit_(3, d3, lee, extra, root, 1024, t0, False, evals, root_eval)
        except Exception as exc:  # noqa: BLE001
            out(f"info string Leela unavailable ({exc}): Maia only")
            lc0 = None
    if not stop_flag.is_set() and maia is not None:   # idle time: Maia's line of the positions that can follow, ready before the user steps forward
        PREFETCH_ABORT.clear()
        threading.Thread(target=prefetch_maia, args=(b, best_dist), daemon=True).start()
    if infinite:  # go infinite: keep the display until the stop command, like a classic engine
        while not stop_flag.is_set():
            time.sleep(0.1)
    out(f"bestmove {pick_move(best_dist)}")


def setpos(tokens):
    global board
    if tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens[0] == "fen":
        i = tokens.index("moves") if "moves" in tokens else len(tokens)
        board = chess.Board(" ".join(tokens[1:i]))
        rest = tokens[i:]
    else:
        return
    if rest and rest[0] == "moves":
        for mv in rest[1:]:
            try:
                board.push_uci(mv)
            except Exception:  # noqa: BLE001
                break


# ----------------------------------------------------------------------------------------------------- UCI options (English; old French names accepted as aliases)
OPTION_ALIASES = {"Elo": "Rating", "Source du Elo": "Rating Source", "Mode de jeu": "Time Control", "Affichage": "Display", "Coup joue": "Mode", "Move Choice": "Mode",
                  "Temps": "Thinking Time"}
VALUE_ALIASES = {"Lichess classique": "Lichess Classical", "Lichess rapide": "Lichess Rapid", "Lichess blitz": "Lichess Blitz", "chess.com rapide": "Chess.com Rapid",
                 "chess.com blitz": "Chess.com Blitz", "Classique": "Classical", "Rapide": "Rapid", "Rapide et Blitz": "Blitz"}


def uci_options():
    out("option name White Rating type spin default 0 min 0 max 3600")
    out("option name Black Rating type spin default 0 min 0 max 3600")
    out("option name Rating Source type combo default FIDE " + " ".join(f"var {s}" for s in SOURCES))
    out("option name Time Control type combo default Classical " + " ".join(f"var {s}" for s in TIME_CONTROLS))
    out(f"option name Display type combo default {DEFAULT_DISPLAY} " + " ".join(f"var {s}" for s in DISPLAYS))
    out("option name MultiPV type spin default 500 min 1 max 500")
    out("option name UCI_ShowWDL type check default true")
    # the "Mode" option (Prediction | Human) is deliberately not advertised: it only matters for a bot build


def set_option(t):
    i = t.index("name")
    j = t.index("value") if "value" in t else len(t)
    nm, val = " ".join(t[i + 1:j]), " ".join(t[j + 1:])
    nm = OPTION_ALIASES.get(nm, nm)
    val = VALUE_ALIASES.get(val, val)
    if nm == "Rating":   # the old single-rating option no longer exists: say so instead of ignoring it silently
        out("info string Toby: the option Rating no longer exists, use White Rating and Black Rating (one filled = both sides get it)")
    elif nm == "MultiPV":
        opts["MultiPV"] = int(float(val))
    elif nm == "White Rating":
        opts["white_rating"] = max(0, int(float(val)))
    elif nm == "Black Rating":
        opts["black_rating"] = max(0, int(float(val)))
    elif nm == "Thinking Time":
        pass  # accepted for compatibility: the search always runs to completion (1024 nodes, complements, evaluation)
    elif nm == "UCI_ShowWDL":
        opts["UCI_ShowWDL"] = val.lower() == "true"
    elif nm == "Rating Source" and val in SOURCES:
        opts["source"] = val
    elif nm == "Time Control" and val in TIME_CONTROLS:
        opts["mode"] = val
    elif nm == "Mode":
        opts["move_choice"] = "human" if val.lower().startswith("human") else "prediction"
    elif nm == "Display" and val in DISPLAYS:
        opts["display"] = val


def main():
    global worker
    threading.Thread(target=load_maia, daemon=True).start()
    threading.Thread(target=preload_lc0, daemon=True).start()
    for line in sys.stdin:
        if LOG:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(f"{_stamp()} > " + line)
        t = line.strip().split()
        if not t:
            continue
        c = t[0]
        if c == "uci":
            out(f"id name {NAME}")
            out("id author Yosha")
            uci_options()
            out("uciok")
        elif c == "isready":
            ready.wait()
            out("readyok")
        elif c == "setoption":
            try:
                set_option(t)
            except Exception:  # noqa: BLE001
                pass
        elif c == "ucinewgame":
            stop_flag.set()
            PREFETCH_ABORT.set()
        elif c == "position":
            PREFETCH_ABORT.set()
            setpos(t[1:])
        elif c == "go":
            PREFETCH_ABORT.set()
            if worker is not None and worker.is_alive():   # a new search never starts while the previous one still runs (it stops within about 50 ms)
                stop_flag.set()
                worker.join(timeout=3)
            stop_flag.clear()
            mt = None
            if "movetime" in t:
                mt = int(t[t.index("movetime") + 1]) / 1000
            worker = threading.Thread(target=search, args=(mt, "infinite" in t), daemon=True)
            worker.start()
        elif c == "stop":
            stop_flag.set()
        elif c == "quit":
            stop_flag.set()
            break
    if lc0 is not None:
        lc0.quit()


if __name__ == "__main__":
    main()
