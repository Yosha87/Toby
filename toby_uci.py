# -*- coding: utf-8 -*-
r"""
toby_uci.py - Toby, The Oracle By Yosha: UCI engine (see README.md and docs/METHOD.md).

Idea: Maia-3 gives the move distribution of a player of strength E; Leela (lc0) gives the value of each move; Toby corrects Maia.
Default since 2026-10-20 (Move Model = Auto): a small neural network trained on real games (nn/corrector.py, nn/README.md), validated on a sealed set
in the three time controls (the protocol of that test, the method page). Move Model = Rule gives the earlier hand-made rule:
    score = log p - lambda * loss/100 + gamma * [Leela's best move] (+ a minimum prior for that move from E = 2200)      then softmax
  Classical: Maia at min(E+700, 2900); lambda = max(0, 3.44 + 5.25 (E-2000)/1000); gamma = 0.43
  Rapid:     Maia at min(2800, Lichess-rapid equivalent of E); lambda = max(0, (E-2100)/72.5); gamma = 0.43 min(1, lambda)
  Blitz:     Maia at min(2800, Lichess-blitz equivalent of E); lambda = max(0, (E-2200)/98);   gamma = 0.26 min(1, lambda)
  (validated on sealed game sets, see docs/METHOD.md; below its threshold Toby plays like Maia.)
Default since 2026-10-26 (option Precision = Accurate, i.e. Calculator = Auto; Move Model = Auto; Precision = Fast = Stockfish always; sealed test v2, the protocol of that test, the method page): Maia given the
previous positions; Classical with an NVIDIA GPU: Leela + Stockfish complements + the history network (L-Leela); elsewhere the Stockfish calculator
(L-SF). In Classical a Maia-3 fine-tuned on over-the-board classical games (maia/maia3-79m-ft-classical.pt) answers, at E+700 without the cap,
read by networks retrained for it (option Fine-tuned Maia). Move Model = Network and Calculator = Leela give the 2026-10-20 default (network v1).
E = FIDE-equivalent strength of the rating entered (ChessGoals tables, toby_rating_convert.py).

Display: 1) Maia alone (instant), 2) Leela's static value of the plausible moves, 3) Leela 1024 nodes, then 64-node complements for the moves it did not explore, then the final distribution
and Toby's win/draw/loss evaluation of each move (davidson model, six sets of constants, see wdl_model).
Ratings: the options White Rating / Black Rating (0 = not set). Both set: each side has its own rating; the side to move is the "player", the other one the opponent, whose rating Maia receives
(moves: policy; evaluation: value head and an explicit rating-difference term). Only one set: both sides get that rating (the opponent is assumed to have the player's rating, the validated setting).
None set: both sides get 2000.
UCI options: White Rating, Black Rating, Rating Source, Time Control, Display (ChessBase | Nibbler | Standard), MultiPV, UCI_ShowWDL.
Default paths can be changed with the environment variables TOBY_LC0, TOBY_NET, TOBY_MAIA_REPO, TOBY_DEVICE.
"""
import atexit
import json
import math
import os
import queue
import random
import shutil
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
VERSION = "2.0"   # public name "Toby 2", version 2.0 (Yosha, 2026-11-22) = internal 2.2.2: 2.2.1 (Toby 2.2 without Toby 1.0 / Maia) + the E_A display lines (the version Siby reads in toby_manifest.json and in Toby/VERSION)


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
LC0 = os.environ.get("TOBY_LC0") or next((str(c) for c in (HERE / "lc0" / "lc0.exe", HERE / "lc0.exe", HERE / "lc0" / "lc0", HERE / "lc0bin") if c.is_file()), "lc0")   # next to Toby (Windows or Linux), else on the PATH, or TOBY_LC0
NET = os.environ.get("TOBY_NET") or next((str(c) for c in sorted((HERE / "nets").glob("*.pb.gz")) + sorted(HERE.glob("*.pb.gz"))), str(HERE / "nets" / "BT3-768x15x24h-swa-2790000.pb.gz"))
MAIA_REPO = os.environ.get("TOBY_MAIA_REPO", str(HERE / "maia3"))   # a clone of https://github.com/CSSLab/maia3
DEVICE = os.environ.get("TOBY_DEVICE", "auto")   # auto: Maia on the NVIDIA GPU when torch sees one, else on the processor
MIN_N = 8  # a move needs this many Leela visits for its value to be trusted without a complementary search
LOG = os.environ.get("TOBY_LOG")

SOURCES = ["FIDE", "Lichess Classical", "Lichess Rapid", "Lichess Blitz", "Chess.com Rapid", "Chess.com Blitz"]
TIME_CONTROLS = ["Classical", "Rapid", "Blitz"]
DISPLAYS = ["ChessBase", "Nibbler", "Standard", "Full"]
DEFAULT_RATING = 2000   # used when neither White Rating nor Black Rating is set
opts = {"white_rating": 0, "black_rating": 0, "source": "FIDE", "mode": "Classical", "display": DEFAULT_DISPLAY, "move_choice": "prediction",
        "MultiPV": 500, "UCI_ShowWDL": True,
        "calculator": os.environ.get("TOBY_CALCULATOR", "Auto"), "sf_threads": int(os.environ.get("TOBY_SF_THREADS", "0")), "sf_nodes": 0,
        "maia_ft": os.environ.get("TOBY_MAIA_FT_ON", "1") == "1", "maia_threads": int(os.environ.get("TOBY_TORCH_THREADS", "0")),
        "hybrid": False, "wdl_endgame": "M", "deepening": True}   # Precision = Hybrid (sealed test v5, amendment 4): Stockfish V2 + Leela's raw policy, network H8

if os.name == "nt" and (sys.stdin is None or sys.stdout is None):   # windowed build (PyInstaller --noconsole): open the interface's pipes ourselves
    try:
        import ctypes
        import msvcrt
        _k32 = ctypes.windll.kernel32
        _k32.GetStdHandle.restype = ctypes.c_void_p
        if sys.stdin is None:
            sys.stdin = open(msvcrt.open_osfhandle(_k32.GetStdHandle(-10), os.O_RDONLY), "r", encoding="utf-8", errors="replace")
        if sys.stdout is None:
            sys.stdout = open(msvcrt.open_osfhandle(_k32.GetStdHandle(-11), os.O_WRONLY), "w", encoding="utf-8", buffering=1)
    except Exception:  # noqa: BLE001
        pass
def _quiet_sink():
    """Where stray output goes when there is no console (PyInstaller windowed build): TOBY_LOG (a file, appended) or the null device."""
    try:
        return open(os.environ.get("TOBY_LOG") or os.devnull, "a", encoding="utf-8", buffering=1)
    except Exception:  # noqa: BLE001
        return open(os.devnull, "w", encoding="utf-8")


def _stream_ok(f):
    """A stream Python can really write to: not None, and its file descriptor open (a GUI-subsystem parent, e.g. a Tcl pipe from
    pythonw / wish, can give the child an invalid or closed stderr handle)."""
    if f is None:
        return False
    try:
        os.fstat(f.fileno())
        return True
    except Exception:  # noqa: BLE001
        return False


class _SafeStream:
    """stderr as seen by libraries (warnings, logging, tqdm, torch, Maia's 'loading checkpoint' prints): writes that fail are dropped and
    the stream switches to the null device, so that no stray write can raise in a search thread."""

    def __init__(self, f):
        self._f = f

    def write(self, x):
        try:
            return self._f.write(x)
        except Exception:  # noqa: BLE001
            try:
                self._f = open(os.devnull, "w", encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            return len(x) if isinstance(x, str) else 0

    def flush(self):
        try:
            self._f.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self):
        return False

    def __getattr__(self, k):
        return getattr(self._f, k)


if not _stream_ok(sys.stderr) or os.environ.get("TOBY_LOG"):   # windowed build or an invalid stderr; or the user asked for a log file
    sys.stderr = _quiet_sink()
    try:   # stray C-level output of libraries (lc0 is a separate process; torch / onnxruntime warnings) goes there too
        os.dup2(sys.stderr.fileno(), 2)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":   # child processes (Stockfish, lc0, nvidia-smi) inherit a valid stderr handle, not the invalid one
        try:
            import ctypes as _ct
            import msvcrt as _ms
            _ct.windll.kernel32.SetStdHandle(-12, _ct.c_void_p(_ms.get_osfhandle(sys.stderr.fileno())))
        except Exception:  # noqa: BLE001
            pass
if not _stream_ok(sys.stdout):   # no UCI stream at all (started without pipes): never crash on a write
    sys.stdout = _quiet_sink()
sys.stderr = _SafeStream(sys.stderr)
try:
    import logging as _logging
    _logging.raiseExceptions = False   # a failing log handler never raises into the engine
except Exception:  # noqa: BLE001
    pass
REAL_OUT = sys.stdout  # the UCI stream; stray prints from libraries (Maia) go to stderr
sys.stdout = sys.stderr
lock = threading.Lock()
stop_flag = threading.Event()
board = chess.Board()
worker = None
PRIOR = {}  # Maia's probabilities before Toby's correction (shown as P in Nibbler)
EV_LAST = {}  # last evaluation of each move in the current search (see emit)
EVAL_K = [None]   # number of moves the current search evaluates (Fast: SF_EVAL_MOVES; Accurate with Leela: 30): no more are listed


def heartbeat_loop():
    """During a search, 'info string toby alive <elapsed ms>' after each second without any line (until the next line or bestmove): a
    GUI can tell a long search from a hung engine. Display only."""
    while True:
        time.sleep(0.1)
        t0 = HEART["t0"]
        if t0 is not None and time.time() - HEART["last"] >= 1.0:
            out(f"info string toby alive {int((time.time() - t0) * 1000)}")
ROOT_LAST = []  # last evaluation (wl, d) of the position itself in the current search (see emit)
PLAY_LOSS = {}   # Mode Human, Fast, Classical: Stockfish's expected-score loss (points) of each move of the current search, and the mover's strength
# Play realism (Mode Human only, never the probabilities shown): the move drawn uses p(m) * exp(-lambda(E) * loss(m)), loss = Stockfish's
# expected-score loss in points (the Fast network's own input). Without it, Fast drew about twice the real blunder rate from 2000 up (open v3
# positions: 2.30% vs 1.06% at 2400+). lambda fitted on the open v3 positions, checked on sealed v5 (not used to fit): blunders 2.07% -> 1.01%
# at 2400+ (real 1.07%), unchanged below 2000. the development log, 2026-11-04.
PLAY_LAMBDA = [(2000, 0.0), (2100, 0.035), (2300, 0.045), (2475, 0.075), (2650, 0.10)]


def play_lambda(E):
    if os.environ.get("TOBY_PLAY_REALISM", "0") != "1":   # internal test tool only (off by default): the proper fix is the retrained networks
        return 0.0
    xs = [a for a, _ in PLAY_LAMBDA]
    if E <= xs[0]:
        return PLAY_LAMBDA[0][1]
    if E >= xs[-1]:
        return PLAY_LAMBDA[-1][1]
    for (x0, y0), (x1, y1) in zip(PLAY_LAMBDA, PLAY_LAMBDA[1:]):
        if x0 <= E <= x1:
            return y0 + (y1 - y0) * (E - x0) / (x1 - x0)
    return 0.0
LISTED = set()  # moves already listed to Nibbler in the current search
T2_SHOWN = set()  # moves Toby 2 evaluated in the current search (Move Evaluations = Toby 2): all listed, also below 0.1 % (2.0.2)
maia = None
maia_ft = None   # Maia-3 fine-tuned on over-the-board classical games (nn/maia_ft/), used in Classical with the networks trained for it (see ft_on)
MAIA_FT_PATH = os.environ.get("TOBY_MAIA_FT") or next((str(c) for c in (HERE / "maia" / "maia3-79m-ft-classical.pt", HERE / "nn" / "maia3_ft.pt")
                                                    if c.exists()), str(HERE / "maia" / "maia3-79m-ft-classical.pt"))
lc0 = None
PREFETCH_ABORT = threading.Event()
ready = threading.Event()
ft_loaded = threading.Event()   # the fine-tuned Maia's loading is over (loaded, absent or failed)
load_error = None


def _stamp():
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(1000 * (t % 1)):03d}"


HEART = {"t0": None, "last": 0.0}   # heartbeat of a search: its start, the time of the last line sent (see heartbeat_loop)


_MUTE = threading.local()   # option Deepening: a nested evaluation of a next position prints nothing (_MUTE.on), its blocks are captured


def out(s):
    if getattr(_MUTE, "on", False):
        return
    with lock:
        REAL_OUT.write(s + "\n")
        REAL_OUT.flush()
        HEART["last"] = time.time()
        if s.startswith("bestmove"):
            HEART["t0"] = None
            BEST_SENT[0] += 1
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


RELEASE_FILES = ("corrector_sf_ft_level2_weights.json", "maia_ft_rating_map_candidate2.json", "release_v6.json", "wdl_ext_params_v4.json", "wdl_mix_q_classical.json", "deepening_params.json",
                 "corrector_hyb_v6b_weights.json", "corrector_sf_weights.json",
                 "corrector_acc_v6aa_weights.json")   # networks of the release (nn/release_v6.json) and the Stockfish-only fallback
_FT_MAP = None   # learnt map E -> Maia input of the fine-tuned Maia (nn/level/fit_rating_map.py): candidate T5, adopted by sealed test v5
try:
    import json as _json_map
    _FT_MAP = _json_map.loads(Path(os.environ.get("TOBY_MAIA_FT_MAP") or (HERE / "nn" / "maia_ft_rating_map_candidate2.json")).read_text(encoding="utf-8"))
except Exception:  # noqa: BLE001   (no map file: E + 700, the earlier rule)
    _FT_MAP = None


def maia_ft_input(E):
    """Maia input of the fine-tuned Maia for a FIDE strength E: the learnt map (nn/maia_ft_rating_map_candidate2.json, linear between its
    points, slope 1 outside), or E + 700 without it."""
    fmap = _FT_MAP
    if fmap is None:
        return E + 700
    es, xs = fmap["E"], fmap["x"]
    if E <= es[0]:
        return xs[0] - (es[0] - E)
    if E >= es[-1]:
        return xs[-1] + (E - es[-1])
    for k in range(1, len(es)):
        if E <= es[k]:
            return xs[k - 1] + (xs[k] - xs[k - 1]) * (E - es[k - 1]) / (es[k] - es[k - 1])
    return xs[-1]


def rule_params(E):
    """(Maia input rating, lambda, gamma) for the current time control (rule form, lambda, gamma and the Maia input: validated on sealed sets, Classical, Rapid v1 and Blitz v1, 2026-10-06)."""
    mode = opts["mode"]
    if mode == "Classical":
        z = (E - 2000) / 1000
        if ft_on():   # the fine-tuned Maia was trained on E + 700 without the 2900 cap (or a learnt rating map, TOBY_MAIA_FT_MAP)
            return int(round(maia_ft_input(E))), max(0.0, 3.44 + 5.25 * z), 0.43
        return int(round(min(E + 700, 2900))), max(0.0, 3.44 + 5.25 * z), 0.43
    if mode == "Rapid":  # lambda = (E-2100)/72.5; gamma = 0.43 min(1, lambda); Maia at the Lichess-rapid equivalent, capped at 2800
        x = int(round(min(2800.0, from_fide(E, "lichess", "rapid"))))
        lam = max(0.0, (E - 2100) / 72.5)
        return x, lam, 0.43 * min(1.0, lam)
    x = int(round(min(2800.0, from_fide(E, "lichess", "blitz"))))  # Blitz: lambda = (E-2200)/98; gamma = 0.26 min(1, lambda)
    lam = max(0.0, (E - 2200) / 98)
    return x, lam, 0.26 * min(1.0, lam)


# ----------------------------------------------------------------------------------------------------- Maia
_LC0_START = threading.Lock()   # one lc0 only: the background start and a search never start two


def ensure_lc0(light=False):
    """The lc0 process, started now if it is not running yet (waits for a start already under way). light: only one-node evaluations
    will be asked (Fast's raw policy): lc0 starts with the smallest footprint (Lc0(light=True)); a full search later restarts it."""
    global lc0
    with _LC0_START:
        if lc0 is not None and lc0.light and not light:   # a search (Accurate) after the light start: the full lc0 replaces it
            try:
                lc0.quit()
                _LC0_PROCS.remove(lc0)
            except Exception:  # noqa: BLE001
                pass
            lc0 = None
        if lc0 is None:
            lc0 = Lc0(light=light)
    return lc0


_LEELA_STARTING = [None]   # the thread of the lazy start of lc0 (one at a time)


def start_leela_for_search():
    """At a 'go' that will use Leela (Accurate with the Leela calculator: the search; Fast hybrid: one Leela evaluation per move), start lc0
    in the background now, so that it loads while the search's Maia stage runs. Never at isready, never for 'toby likelihood' / 'dist' /
    batches (they never use Leela). The search itself waits for the same start (one lc0, one lock): the values are those of a cold start."""
    if lc0 is not None or LEELA_BROKEN or (_LEELA_STARTING[0] is not None and _LEELA_STARTING[0].is_alive()):
        return
    if t2_on():   # Toby 2: no Leela program at all (network B replaces Maia + lc0 for the move distribution)
        return
    target = preload_lc0 if calculator() == "Leela" else (preload_raw_policy if hybrid_on() else None)
    if target is not None:
        _LEELA_STARTING[0] = threading.Thread(target=target, daemon=True)
        _LEELA_STARTING[0].start()


def preload_lc0(leela=True):
    """Stockfish calculator: start and warm up Stockfish. Leela calculator (leela=True): start lc0 (not at program start: only once the
    GUI's options are known, at the first search that needs it, start_leela_for_search)."""
    global lc0, sf_engine
    if calculator() == "Stockfish":   # Stockfish started (and warmed up) before the first search
        try:
            sf_engine = StockfishCalc()
            sf_engine.evaluate(chess.Board(), 20000)
        except Exception:  # noqa: BLE001
            sf_engine = None
        return
    if calculator() != "Leela" or not leela:
        return
    try:
        ensure_lc0()
        if calculator() != "Leela":   # switched to Stockfish while lc0 was starting
            stop_all_lc0()
            lc0 = None
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


CLOCK = None   # (seconds on the mover's clock, on the opponent's) of the current position, from 'go wtime btime', or None (unknown)


def clock_feats(me, opp):
    """The 3 time inputs of a Maia time model (same as nn/maia_ft/data.py): 10 x [known, f(mover), f(opponent)], f(s) = log(1 + s/60) / log(121)."""
    if me is None or opp is None:
        return [0.0, 0.0, 0.0]
    f = lambda x: math.log1p(max(0.0, x) / 60.0) / math.log(121.0)
    return [10.0, 10.0 * f(me), 10.0 * f(opp)]


def _load_maia_weights(eng, torch, path):
    """Loads fine-tuned weights over the released Maia-3; a time model (3 more input columns, nn/maia_ft/data.py widen_for_time) is
    recognised by the width of its token projection and the model is widened first."""
    sd = torch.load(path, map_location=eng.cfg.device)
    m = eng.model
    tp = m.token_projection
    if sd["token_projection.weight"].shape[1] == tp.in_features + 3:
        new = torch.nn.Linear(tp.in_features + 3, tp.out_features).to(device=tp.weight.device, dtype=tp.weight.dtype)
        m.token_projection = new
        m.cfg.include_time_info = True
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    m.eval()


def is_time_model(model):
    return model is not None and bool(getattr(model[0].model.cfg, "include_time_info", False))


def current_tf(model, clock="current"):
    """Time inputs given to `model` (None for a model without time inputs); clock: (mover s, opponent s), None, or "current" (CLOCK)."""
    if not is_time_model(model):
        return None
    c = CLOCK if clock == "current" else clock
    return tuple(round(x, 4) for x in clock_feats(*(c or (None, None))))


def mp_key(fens, elo, oppo, is_ft, tf=None):
    """Key of the Maia policy cache _MP (the time inputs only for a time model, so that other keys are unchanged)."""
    return (fens, elo, oppo, is_ft) if tf is None else (fens, elo, oppo, is_ft, tf)


def time_tokens(eng, tokens, tf):
    """Tokens [B, 64, D] of a time model: the 3 time inputs after the board planes (unknown clock: zeros); unchanged otherwise."""
    if not getattr(eng.model.cfg, "include_time_info", False):
        return tokens
    import torch
    h = 12 * eng.model.cfg.history
    tokens = tokens[:, :, :h + 4].clone()
    tokens[:, :, h:h + 3] = torch.tensor(tf or (0.0, 0.0, 0.0), dtype=tokens.dtype, device=tokens.device)
    tokens[:, :, h + 3] = 0.0
    return tokens


def maia_device():
    """TOBY_DEVICE (cpu / cuda), or with auto (default) the NVIDIA GPU when torch has CUDA and sees one, else the processor."""
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def maia_args():
    """Maia-3 arguments: the released checkpoint given by TOBY_MAIA_WEIGHTS, shipped next to Toby (maia/maia3-79m.pt), or already in the
    Hugging Face cache is given directly, so that no request goes to the Hugging Face Hub (offline use, faster start); else Maia downloads it."""
    args = ["--model", "maia3-79m", "--device", maia_device(), "--no-use-amp"]
    c = maia_checkpoint()
    return args + ["--checkpoint", c] if c else args


def maia_checkpoint():
    """The released Maia-3 checkpoint maia_args gives (shipped next to Toby, TOBY_MAIA_WEIGHTS, or the Hugging Face cache), or None
    (Maia would then download it). No torch import: usable at 'uci'."""
    for c in (os.environ.get("TOBY_MAIA_WEIGHTS"), HERE / "maia" / "maia3-79m.pt", HERE / "nn" / "maia3-79m.pt"):   # shipped with the package
        if c and Path(c).is_file():
            return str(c)
    try:
        roots = [os.environ.get("HF_HUB_CACHE"), os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
                 os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
        for r in roots:
            if r:
                snaps = sorted(Path(r, "models--UofTCSSLab--Maia3-79M", "snapshots").glob("*/maia3-79m.pt"), key=lambda q: q.stat().st_mtime)
                if snaps:
                    return str(snaps[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def set_maia_threads(n):
    """Torch threads of this process (option Maia Threads; 0 = torch's default, all cores). Set it when several Tobys run side by side."""
    if n and n > 0:
        try:
            import torch
            torch.set_num_threads(int(n))
        except Exception:  # noqa: BLE001
            pass


STARTUP = {"t0": time.time()}   # startup timings, printed as 'info string toby timing ...' (GUIs measure loading on the user's machine)


def load_maia():
    global maia, load_error
    try:
        if MAIA_REPO not in sys.path:
            sys.path.insert(0, MAIA_REPO)
        import torch
        STARTUP["torch"] = time.time()
        from maia3.dataset import get_legal_moves_mask
        from maia3.uci import Maia3UCIEngine, parse_args
        set_maia_threads(opts["maia_threads"])
        cfg = parse_args(maia_args())
        eng = Maia3UCIEngine(cfg)
        eng.ensure_model_loaded()
        if os.environ.get("TOBY_MAIA_CKPT"):   # fine-tuned Maia-3 weights (nn/maia_ft/train_maia.py), loaded over the released ones
            _load_maia_weights(eng, torch, os.environ["TOBY_MAIA_CKPT"])
        maia = (eng, torch, get_legal_moves_mask)
        STARTUP["maia"] = time.time()
    except Exception as exc:  # noqa: BLE001
        load_error = f"Maia: {exc}"
    ready.set()
    try:
        if maia is not None:   # after ready: Rapid and Blitz answer at once; a Classical search waits for this loading (wait_maia_ft)
            _load_maia_ft(torch, Maia3UCIEngine, parse_args, get_legal_moves_mask)
            STARTUP["ft"] = time.time()
    finally:
        ft_loaded.set()


_MAIA_START = {"started": False, "lock": threading.Lock()}


def _toby1_shipped():
    """2.2.1: Toby 1.0 (Maia-3, the fine-tuned Maia and the Toby 1.0 engine) is part of this copy only if Maia-3's weights ship with it:
    maia/maia3-79m.pt (or nn/maia3-79m.pt) next to Toby, or TOBY_MAIA_WEIGHTS. A Maia-3 found only in the user's Hugging Face cache does
    not count. TOBY_TOBY1=0 / 1 forces it (development). False: no code path loads Maia (Toby 2 answers everything, said)."""
    v = os.environ.get("TOBY_TOBY1")
    if v in ("0", "1"):
        return v == "1"
    for c in (os.environ.get("TOBY_MAIA_WEIGHTS"), HERE / "maia" / "maia3-79m.pt", HERE / "nn" / "maia3-79m.pt"):
        if c and Path(c).is_file():
            return True
    return False


TOBY1_SHIPPED = _toby1_shipped()
T1_ANSWERS = "Toby 1.0 answers" if TOBY1_SHIPPED else "fallback answer below (no Toby 1.0 in this Toby)"
TOBY1_ABSENT = "Toby 1.0 is not shipped with this Toby (Maia-3 left out): use 'toby level'"


def ensure_maia():
    """Starts loading Maia-3 (and then the fine-tuned Maia) in the background, once per process. Since 2.0.2 Maia is loaded only when
    something uses it: 'toby likelihood' / 'toby batch' (Siby's level estimate), a Toby 1.0 search, or a Toby 2 search with a Toby 1.0
    module (Move Evaluations = Toby 1.0, Position Evaluation = Toby 1.0 / Mix, Candidates = Maia C). A Toby 2 search with the default
    options never uses Maia. Whoever needs Maia then waits on `ready` (set when the loading ends, loaded or failed)."""
    global load_error
    with _MAIA_START["lock"]:
        if _MAIA_START["started"]:
            return
        _MAIA_START["started"] = True
    if not TOBY1_SHIPPED:   # never loads Maia: whoever waits on `ready` sees maia = None at once (load_error says why)
        load_error = TOBY1_ABSENT
        ready.set()
        ft_loaded.set()
        return
    threading.Thread(target=load_maia, daemon=True).start()


def maia_wanted():
    """True when the current options make the next search use Maia (see ensure_maia); 'toby likelihood' asks for it itself."""
    if not TOBY1_SHIPPED:
        return False
    if os.environ.get("TOBY_MAIA_EAGER") == "1" or not t2_on():
        return True
    cand_maia = opts["t2_candidates"] == "Maia C" and (opts.get("t2_candidates_set") or T2.get("cfg") is not None
                                                       or t2_config_path() is None)
    return opts["t2_move_evals"] == "Toby 1.0" or opts["t2_position_eval"] in ("Toby 1.0", "Mix") or cand_maia


def wait_maia(ft=False, timeout=None):
    """ensure_maia, then wait for the loading (and for the fine-tuned Maia when ft and Classical wants it)."""
    ensure_maia()
    if not ready.is_set():
        ready.wait(timeout)   # timeout: a Toby 2 search with a Toby 1.0 module waits at most this long (then the module is Off, said)
    if ft:
        wait_maia_ft()


def wait_maia_ft():
    """Classical with the fine-tuned Maia installed and wanted: wait (at most 3 min) until it is loaded, so that the very first moves already use it."""
    if opts["mode"] == "Classical" and opts.get("maia_ft", True) and os.path.exists(MAIA_FT_PATH) and not ft_loaded.is_set():
        out("info string Toby: loading the fine-tuned Maia (Classical)...")
        ft_loaded.wait(180)


def _load_maia_ft(torch, Maia3UCIEngine, parse_args, get_legal_moves_mask):
    """Second Maia-3 with the fine-tuned weights (MAIA_FT_PATH), when the file and a network trained for it exist; Maia stays usable without it."""
    global maia_ft
    if not os.path.exists(MAIA_FT_PATH) or (CORRECTOR_SF_FT is None and CORRECTOR_HSF_FT is None):
        return
    try:
        import copy
        cfg = parse_args(maia_args())
        eng = Maia3UCIEngine(cfg)
        base = maia[0] if maia is not None else None
        if base is not None and base.model is not None and not getattr(base.model.cfg, "include_time_info", False):
            eng.model = copy.deepcopy(base.model)   # same architecture: copy the loaded model instead of reading the checkpoint again
            eng.cfg.checkpoint_path = base.cfg.checkpoint_path
        else:
            eng.ensure_model_loaded()
        _load_maia_weights(eng, torch, MAIA_FT_PATH)
        maia_ft = (eng, torch, get_legal_moves_mask)
    except Exception as exc:  # noqa: BLE001
        out(f"info string Toby: fine-tuned Maia not loaded ({exc})")
        maia_ft = None


_FT = threading.local()   # _FT.force: True / False overrides ft_on() in this thread (the likelihood command uses the Stockfish network whatever the calculator)


# ---------------------------------------------------------------------------------------------- reused worker threads (2.2.1, Windows leak)
# On Windows, each new OS thread that runs torch keeps native per-thread buffers after it ends (Siby, 2026-11-21: ~8.6 MB of private
# bytes per 'toby level move', never released; flat on Linux). Every command and every search used to start new threads (here and in
# toby2.py). PooledThread has threading.Thread's interface (start / join / is_alive / daemon / name) but runs its target on a worker
# thread that is kept and reused, so the number of OS threads stays at the peak concurrency. The targets, their order and their
# results are unchanged; the worker's thread-local state (_MUTE, _FT) is cleared after each job. Default on Windows only (below).
_POOL_LOCK = threading.Lock()
_POOL_IDLE = []


class _PoolWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="toby-pool")
        self.jobs = queue.SimpleQueue()
        self.start()

    def run(self):
        while True:
            job = self.jobs.get()
            try:
                job._run()
            finally:
                for tl in (_MUTE, _FT):
                    tl.__dict__.clear()
                with _POOL_LOCK:   # idle again BEFORE the job reads as finished: a start() right after join() reuses this worker
                    _POOL_IDLE.append(self)
                job._done.set()


class PooledThread:
    """threading.Thread look-alike whose target runs on a reused worker thread (see above)."""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self._target, self._args, self._kwargs = target, tuple(args), dict(kwargs or {})
        self._done, self._started = threading.Event(), False
        self.name, self.daemon = name or "pooled", True if daemon is None else daemon

    def start(self):
        if self._started:
            raise RuntimeError("threads can only be started once")
        self._started = True
        with _POOL_LOCK:
            w = _POOL_IDLE.pop() if _POOL_IDLE else None
        (w or _PoolWorker()).jobs.put(self)

    def _run(self):
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        except BaseException:  # noqa: BLE001   (as threading.Thread: the traceback goes to stderr, the worker lives on)
            import traceback
            traceback.print_exc()

    def join(self, timeout=None):
        if not self._started:
            raise RuntimeError("cannot join thread before it is started")
        self._done.wait(timeout)

    def is_alive(self):
        return self._started and not self._done.is_set()


# default: on Windows only (Linux keeps 2.2's plain threads: no leak there, and the reused threads hold ~100 MB more of malloc arenas, measured
# on 400 go / stop cycles); TOBY_THREAD_POOL=1 / 0 forces it
WThread = PooledThread if os.environ.get("TOBY_THREAD_POOL", "1" if os.name == "nt" else "0") != "0" else threading.Thread


class _ThreadingForToby2:
    """The `threading` module as toby2.py sees it: the same module, with Thread = WThread (toby2.py's file is unchanged)."""

    def __getattr__(self, k):
        return WThread if k == "Thread" else getattr(threading, k)


def ft_on():
    """True when the fine-tuned Maia answers: Classical, option Fine-tuned Maia on, and a network trained with it for the current calculator
    (Stockfish: corrector_sf_ft_weights.json; Leela with Move Model = History: corrector_hist_sf_ft_weights.json). Elsewhere the released Maia."""
    f = getattr(_FT, "force", None)
    if f is not None:
        return f
    if maia_ft is None or opts["mode"] != "Classical" or not opts.get("maia_ft", True):
        return False
    if calculator() == "Stockfish":
        return CORRECTOR_SF is not None and CORRECTOR_SF_FT is not None
    return hist_on() and CORRECTOR_HSF_FT is not None


def sf_corrector():
    """Stockfish network of the current setting."""
    return CORRECTOR_SF_FT if ft_on() else CORRECTOR_SF


def budget_input(cor, nodes):
    """Node budget given to a Stockfish network: only to one trained with it (a network trained without it has a constant zero column: mean 0, sd 1)."""
    names = cor.meta.get("inputs") or []
    if "sf_budget" not in names:
        return None
    i = names.index("sf_budget")
    return nodes if (cor.mu[i] != 0.0 or cor.sd[i] != 1.0) else None


MAIA_LOCK = threading.RLock()   # the Maia model keeps its board / history inside: one call at a time (search thread and idle-time prefetch)
_MP = {}                          # Maia policy by (position, rating, opponent rating): going back to a position, or stepping to a prefetched one, costs nothing


MAIA_HISTORY = os.environ.get("TOBY_MAIA_HISTORY", "0") == "1"   # give Maia the previous positions of the game (move list of the GUI); see history_boards


def history_boards(b, n=8):
    """The current position and up to n-1 previous ones (oldest first), from the board's move stack (empty stack: the position alone,
    e.g. 'position fen ...' without moves). Maia-3 reads 8 positions (the current one and 7 previous)."""
    out = [b.copy(stack=False)]
    c = b.copy()
    while c.move_stack and len(out) < n:
        c.pop()
        out.insert(0, c.copy(stack=False))
    return out


def maia_policy(b, elo, oppo=None, history=None, clock="current"):
    """Maia's move probabilities {uci: p} for a player of the given rating; oppo = the opponent's rating given to Maia (default: the same rating on both sides).
    history: True = the previous positions of b's move stack are given to Maia, False = the position alone; None = the engine's setting (MAIA_HISTORY).
    clock: (mover's seconds, opponent's seconds) or None, for a Maia time model only; "current" = CLOCK (from the go command)."""
    use_h = MAIA_HISTORY if history is None else history
    hb = history_boards(b) if use_h else [b]
    model = maia_ft if (maia_ft is not None and ft_on()) else maia
    tf = current_tf(model, clock)
    k = mp_key(tuple(x.fen() for x in hb), elo, oppo, model is maia_ft, tf)
    r = _MP.get(k)
    if r is None:
        with MAIA_LOCK:
            r = _maia_policy_raw(b, elo, oppo, hb if len(hb) > 1 else None, model, tf)
        if len(_MP) > 3000:
            _MP.clear()
        _MP[k] = r
    return dict(r)


def maia_batch(b, pairs, history=None, clock="current"):
    """Maia's policies of position b for several (rating, opponent rating) pairs in ONE forward pass, stored in the cache of maia_policy (which
    then answers at once): the rating grid of 'toby likelihood' costs one Maia call instead of one per rating. Same model choice, history and
    clock as maia_policy; the rating inputs are the only thing that changes in the batch."""
    use_h = MAIA_HISTORY if history is None else history
    hb = history_boards(b) if use_h else [b]
    model = maia_ft if (maia_ft is not None and ft_on()) else maia
    if model is None:
        return
    tf = current_tf(model, clock)
    fens = tuple(x.fen() for x in hb)
    todo = [pr for pr in dict.fromkeys(pairs) if mp_key(fens, pr[0], pr[1], model is maia_ft, tf) not in _MP]
    if len(todo) < 2:
        return
    eng, torch, mask_fn = model
    cfg = eng.cfg
    with MAIA_LOCK, torch.no_grad():
        eng.board = b
        eng._reset_history()
        if len(hb) > 1:
            from maia3.dataset import tokenize_board
            eng.history.clear()
            for x in hb[-cfg.history:]:
                eng.history.append(tokenize_board(x))
        tok = time_tokens(eng, eng._tokens_from_history(eng.history).unsqueeze(0).to(cfg.device), tf)
        se = torch.tensor([e for e, _ in todo], dtype=torch.long, device=cfg.device)
        so = torch.tensor([e if o is None else o for e, o in todo], dtype=torch.long, device=cfg.device)
        logits_move, _v, _ = eng.model(tok.expand(len(todo), *tok.shape[1:]), se, so)
        mask = mask_fn(b, eng.all_moves_dict).to(cfg.device)
        probs = torch.softmax(logits_move.float().masked_fill(~mask.unsqueeze(0), float("-inf")), dim=-1).cpu()
        idx = torch.nonzero(mask).flatten().tolist()
        ucis = [(i, eng._move_from_index(i)) for i in idx]
        if len(_MP) > 3000:
            _MP.clear()
        for row, (e, o) in enumerate(todo):
            _MP[mp_key(fens, e, o, model is maia_ft, tf)] = {mv.uci(): float(probs[row, i]) for i, mv in ucis if mv is not None}


def _maia_policy_raw(b, elo, oppo=None, hist=None, model=None, tf=None):
    eng, torch, mask_fn = model or maia
    cfg = eng.cfg
    with torch.no_grad():
        eng.board = b
        eng._reset_history()
        if hist:   # previous positions, oldest first, as Maia-3's own UCI engine does with --use-uci-history
            from maia3.dataset import tokenize_board
            eng.history.clear()
            for x in hist[-cfg.history:]:
                eng.history.append(tokenize_board(x))
        tokens = time_tokens(eng, eng._tokens_from_history(eng.history).unsqueeze(0).to(cfg.device), tf)
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


def _maia_value_raw(b, elo, oppo=None, pol=False):
    eng, torch, mask_fn = maia
    cfg = eng.cfg
    with torch.no_grad():
        eng.board = b
        eng._reset_history()
        tokens = eng._tokens_from_history(eng.history).unsqueeze(0).to(cfg.device)
        se = torch.tensor([elo], dtype=torch.long, device=cfg.device)
        so = se if oppo is None else torch.tensor([oppo], dtype=torch.long, device=cfg.device)
        lm, v, _ = eng.model(tokens, se, so)
        if pol:   # Endgame Evaluation: Maia's policy of the same pass (top probability, entropy over the legal moves)
            _MVP[(b.fen(), elo, oppo)] = _valpass_policy(lm[0], b)
        return torch.softmax(v[0].float(), dim=-1).tolist()


_MVP = {}   # (fen, elo, oppo) -> (top, entropy) of Maia's policy from the value pass (option Endgame Evaluation only)


def _valpass_policy(logits, b):
    eng, torch, mask_fn = maia
    mk = mask_fn(b, eng.all_moves_dict).to(logits.device)
    pr = torch.softmax(logits.float().masked_fill(~mk, float("-inf")), dim=-1)
    return _WE.policy_stats(pr[mk].tolist())


# ----------------------------------------------------------------------------------------------------- Leela
MOVE_RE = re.compile(r"info string (\S+)\s+\(\s*\d+\s*\)\s+N:\s*(\d+).*?\(P:\s*([\d.]+)%\).*?\(WL:\s*([-\d.]+)\).*?\(D:\s*([-\d.]+)\).*?\(Q:\s*([-\d.]+)\)")
ROOT_RE = re.compile(r"info string node.*?\(WL:\s*([-\d.]+)\).*?\(D:\s*([-\d.]+)\)")


_LC0_PROCS = []   # every lc0 started by this Toby: all of them are stopped when Toby ends (quit, closed stdin, normal exit)
_JOB = None       # Windows: a job object that kills lc0 if Toby itself is killed (no quit, no atexit)


_CHILD_PIDS = []  # lc0 and Stockfish processes started by Toby: they get Toby's own priority (at start and at every isready)


def match_priority(pid=None):
    """Give Toby's child processes (one pid, or all of them) the priority of Toby's own process. A GUI that lowers Toby's priority
    (Siby: idle on Windows) cannot reach the engines Toby starts itself; this keeps the interface fluid. No effect on results (fixed node budgets)."""
    pids = [pid] if pid is not None else list(_CHILD_PIDS)
    if pid is not None and pid not in _CHILD_PIDS:
        _CHILD_PIDS.append(pid)
    for q in pids:
        try:
            if os.name == "nt":
                import ctypes
                k32 = ctypes.windll.kernel32
                cls = k32.GetPriorityClass(k32.GetCurrentProcess())
                h = k32.OpenProcess(0x0200, False, int(q))   # PROCESS_SET_INFORMATION
                if h:
                    if cls:
                        k32.SetPriorityClass(h, cls)
                    k32.CloseHandle(h)
            else:
                os.setpriority(os.PRIO_PROCESS, int(q), os.getpriority(os.PRIO_PROCESS, 0))
        except Exception:  # noqa: BLE001
            pass


def _windows_kill_on_close_job():
    """Windows only: a job object with KILL_ON_JOB_CLOSE. Its handle belongs to Toby; when Toby's process ends in any way, Windows closes it
    and ends every process assigned to it (lc0). None elsewhere or on failure."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # 9 = JobObjectExtendedLimitInformation
            return None
        return job, k32
    except Exception:  # noqa: BLE001
        return None


def stop_all_lc0():
    """Ends every lc0 started by Toby: 'quit', then terminate() after one second, then kill(). Safe to call several times."""
    for lc in list(_LC0_PROCS):
        try:
            lc.quit()
        except Exception:  # noqa: BLE001
            pass
    _LC0_PROCS.clear()


atexit.register(stop_all_lc0)


# Engine versions the networks were trained and sealed-tested with (the protocol of that test). Other versions give values on another scale: Toby still
# runs, but its numbers are not validated, and it says so once.
TESTED_SF = "Stockfish 19"
TESTED_LC0 = "Lc0 v0.32"
TESTED_NET = "BT3-768x15x24h-swa-2790000.pb.gz"
_WARNED = set()
VERSIONS = {}   # component -> version found (shown in the summary line of each search)


def _ver(what, name):
    """Comparable version of a component, or None when it cannot be read (then it is not taken as newer)."""
    if what == "Stockfish":
        m = re.match(r"Stockfish (\d+)(?:\.(\d+))?(?:\s|$)", name)
        return tuple(int(x or 0) for x in m.groups()) if m else None
    if what == "Leela":
        m = re.match(r"Lc0 v(\d+)\.(\d+)(?:\.(\d+))?", name)
        return tuple(int(x or 0) for x in m.groups()) if m else None
    m = re.match(r"(BT3-768x15x24h-swa-)(\d+)\.pb(\.gz)?$", name)   # Leela network: the tested family, a later training step is newer
    return (int(m.group(2)),) if m else None


def check_version(what, name, tested):
    """The tested version or a newer one: no message (the version is shown in the summary line of each search). An older one, an unreadable
    one, or another program: a warning, once per process. Toby's numbers are validated with the tested versions only."""
    VERSIONS[what] = name.replace(".pb.gz", "").replace(".pb", "")
    v, t = _ver(what, name), _ver(what, tested)
    if v is not None and t is not None and v >= t:
        return
    if what not in _WARNED:
        _WARNED.add(what)
        out(f"info string Toby: WARNING {what} '{name}' is older than the tested version or not recognized ({tested}): Toby's numbers are not validated with it")


def versions_text():
    return "".join(f"; {VERSIONS[k]}" for k in ("Stockfish", "Leela", "Leela network") if k in VERSIONS)


GPU_BACKENDS = ("cuda", "cudnn", "opencl", "dx12", "onnx-dml", "onnx-trt", "onnx-cuda", "metal", "sycl")


def lc0_backend():
    """The backend Toby gives lc0: TOBY_BACKEND, else toby_backend.txt next to lc0, else cuda with an NVIDIA GPU, else lc0's own default ("")."""
    return os.environ.get("TOBY_BACKEND") or lc0_backend_file() or ("cuda" if HAS_NVIDIA else "")


def lc0_on_gpu():
    """lc0 searches on a graphics card. Accurate (a 1,024-node Leela search per move) needs it: with a CPU lc0 (Toby-light) a search takes
    minutes, so Accurate then takes the Fast (hybrid) path, which needs only one Leela evaluation per move."""
    return lc0_backend().startswith(GPU_BACKENDS)


def lc0_backend_file():
    """The backend written at packaging time next to lc0 (toby_backend.txt: e.g. onnx-cpu, dnnl, blas, cuda-auto): the one that passed the
    policy gate (nn/retrain_v6/policy_gate.py) for that build. None when there is no file."""
    try:
        f = Path(shutil.which(LC0) or LC0).parent / "toby_backend.txt"
        v = f.read_text(encoding="utf-8").strip() if f.exists() else ""
        return v or None
    except Exception:  # noqa: BLE001
        return None


class Lc0:
    """Thin wrapper around an lc0 process (UCI, --verbose-move-stats)."""

    def __init__(self, light=False, backend=None, start_timeout=None, search_timeout=None):
        global _JOB
        self.start_timeout, self.search_timeout = start_timeout, search_timeout   # Toby 2 Accurate: short, fixed (never a long first answer)
        self.pgroup = os.name != "nt" and start_timeout is not None
        self.stop_event = None   # None: searches obey the UCI stop flag; an Event: they obey it instead (the Accurate probe's self-check)
        self._io = threading.RLock()   # one exchange with lc0 at a time (search, static_values, new_game)
        self.light = light
        backend = backend or lc0_backend()   # no NVIDIA GPU, no file: lc0's own default (Toby 2 Accurate passes its product's backend)
        self.backend = backend
        extra = os.environ.get("TOBY_LC0_EXTRA", "").split()
        has = lambda o: any(x.split("=")[0] == o for x in extra)
        if light:   # one-node evaluations only (Fast's raw policy): one search thread, no batching or prefetch, a tiny cache
            for o, v in (("--threads", "1"), ("--minibatch-size", "1"), ("--max-prefetch", "0"), ("--nncache", "2000")):
                if not has(o):
                    extra.append(f"{o}={v}")
        if backend.startswith("cuda") and not has("--backend-opts"):
            # GPU memory ~1.8 GB instead of ~4.2 GB, same speed; results differ only within lc0's run-to-run noise (2026-11-01 test);
            # light: the smallest batch buffers (one position per evaluation)
            extra.append(f"--backend-opts=max_batch={os.environ.get('TOBY_LC0_LIGHT_MAX_BATCH', '8') if light else '256'}")
        self.p = subprocess.Popen([LC0, f"--weights={NET}"] + (["--backend=" + backend] if backend else []) + ["--verbose-move-stats"]
                                  + extra, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),   # Windows: no console window for lc0 when the GUI has none
                                  start_new_session=self.pgroup)   # Toby 2: own process group, so that the watchdog kills a wrapper's children too
        _LC0_PROCS.append(self)
        match_priority(self.p.pid)
        if os.name == "nt":
            if _JOB is None:
                _JOB = _windows_kill_on_close_job() or False
            if _JOB:
                try:
                    _JOB[1].AssignProcessToJobObject(_JOB[0], int(self.p._handle))
                except Exception:  # noqa: BLE001
                    pass
        check_version("Leela network", Path(NET).name, TESTED_NET)
        self.send("uci")
        self.wait("uciok")
        self.new_game()

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _watchdog(self, seconds):
        """Kill lc0 if the current exchange is not over after `seconds` (a GPU that hangs instead of failing): the reading loop then ends
        and the caller raises, so Toby never waits forever."""
        def _kill():
            if getattr(self, "pgroup", False):   # the whole group: a process left holding lc0's output pipe would block the reader forever
                try:
                    import signal
                    os.killpg(self.p.pid, signal.SIGKILL)
                    return
                except Exception:  # noqa: BLE001
                    pass
            if self.p.poll() is None:
                self.p.kill()
        t = threading.Timer(seconds, _kill)
        t.daemon = True
        t.start()
        return t

    def wait(self, token):
        # starting lc0 and loading the network: up to 300 s on a GPU (first CUDA start), 60 s otherwise (a CPU lc0 loads BT3 in seconds;
        # a GPU build without a usable GPU must fail fast)
        dog = self._watchdog(self.start_timeout or float(os.environ.get("TOBY_LC0_START_TIMEOUT", "300" if (HAS_NVIDIA and lc0_on_gpu()) else "60")))
        try:
            self._wait(token)
        finally:
            dog.cancel()

    def _wait(self, token):
        for line in self.p.stdout:
            if line.startswith("id name"):
                check_version("Leela", line[8:].strip(), TESTED_LC0)
            if line.startswith(token):
                return
        raise RuntimeError(f"lc0 ended before '{token}' (exit code {self.p.poll()})")   # never wait forever on a dead lc0

    def new_game(self):
        with self._io:
            self._new_game()

    def _new_game(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait("readyok")

    def search(self, fen, nodes, searchmoves=None):
        """One exchange with lc0 at a time: a background call (the Hybrid preload, leela_raw_policy) and a search must never read each
        other's lines (they would both wait forever)."""
        with self._io:
            return self._search(fen, nodes, searchmoves)

    def _search(self, fen, nodes, searchmoves=None):
        """Returns ({uci: (visits, WL, D, Q)}, (WL, D) of the root). The tree is reset at each call (lc0 would otherwise reuse the tree of the same position)."""
        self.new_game()
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}" + (f" searchmoves {searchmoves}" if searchmoves else ""))
        moves, root, pol = {}, None, {}
        finished = threading.Event()

        stop = self.stop_event or stop_flag   # the Accurate probe passes its own event: 'ucinewgame' / 'stop' set the UCI flag, not this one

        def watch():  # the GUI's stop reaches lc0 at once: the partial result is discarded by the caller, which checks the flag
            while not finished.is_set():
                if stop.wait(0.02):
                    if not finished.is_set():
                        try:
                            self.send("stop")
                        except Exception:  # noqa: BLE001
                            pass
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        dog = self._watchdog(self.search_timeout or float(os.environ.get("TOBY_LC0_SEARCH_TIMEOUT", "120" if nodes <= 1 else "600")))
        got_best = False
        for line in self.p.stdout:
            if line.startswith("bestmove"):
                got_best = True
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
                    pol[m.group(1)] = float(m.group(3)) / 100  # Leela's policy prior, kept for every listed move (also the unvisited ones)
                except ValueError:
                    pass
                try:
                    moves[m.group(1)] = (int(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
                except ValueError:  # unvisited move: values are '-.-----'
                    pass
        finished.set()
        dog.cancel()
        watcher.join()
        if not got_best:
            raise RuntimeError(f"lc0 ended during a search (exit code {self.p.poll()})")
        # lc0 prints a castling move as "king takes own rook" (e1h1, e1a1, e8h8, e8a8); Maia and UCI use e1g1, e1c1, e8g8, e8c8
        try:
            b0 = chess.Board(fen)
            for u in list(moves):
                f, t = chess.parse_square(u[:2]), chess.parse_square(u[2:4])
                if len(u) == 4 and b0.piece_type_at(f) == chess.KING and b0.piece_type_at(t) == chess.ROOK and b0.color_at(f) == b0.color_at(t):
                    moves[chess.square_name(f) + chess.square_name(chess.square(6 if chess.square_file(t) > chess.square_file(f) else 2, chess.square_rank(f)))] = moves.pop(u)
            for u in list(pol):
                f, t = chess.parse_square(u[:2]), chess.parse_square(u[2:4])
                if len(u) == 4 and b0.piece_type_at(f) == chess.KING and b0.piece_type_at(t) == chess.ROOK and b0.color_at(f) == b0.color_at(t):
                    pol[chess.square_name(f) + chess.square_name(chess.square(6 if chess.square_file(t) > chess.square_file(f) else 2, chess.square_rank(f)))] = pol.pop(u)
        except Exception:  # noqa: BLE001
            pass
        self.last_policy = pol  # {uci: Leela prior} of the last search (the return value is unchanged)
        return moves, root

    def static_values(self, b, moves):
        with self._io:
            return self._static_values(b, moves)

    def _static_values(self, b, moves):
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
        """'quit' to lc0, terminate() if it has not left after one second, kill() as a last resort."""
        if self.p.poll() is None:
            try:
                self.send("quit")
            except Exception:  # noqa: BLE001
                pass
            try:
                self.p.wait(timeout=1)
            except Exception:  # noqa: BLE001
                try:
                    self.p.terminate()
                    self.p.wait(timeout=1)
                except Exception:  # noqa: BLE001
                    try:
                        self.p.kill()
                    except Exception:  # noqa: BLE001
                        pass
        if self in _LC0_PROCS:
            _LC0_PROCS.remove(self)


# ----------------------------------------------------------------------------------------------------- Stockfish calculator (Toby-SF)
# Option Calculator = Stockfish: Leela's search is replaced by Stockfish's evaluation of EVERY legal move (MultiPV = number of legal moves, a node budget),
# read by a network trained on exactly these inputs (nn/dataset.py --calc sf, weights nn/corrector_sf_weights.json, Maia with the previous positions).
# No GPU needed for the calculator. Not yet sealed-tested (dev results: the method page).
SF_PATH = os.environ.get("TOBY_STOCKFISH", str(HERE / "stockfish" / ("stockfish.exe" if os.name == "nt" else "stockfish")))
SF_NN_PATH = os.environ.get("TOBY_NN_SF", str(HERE / "nn" / "corrector_sf_weights.json"))
CALCULATORS = ["Auto", "Leela", "Stockfish"]
# Calculator = Auto (default; sealed test v2 of 2026-10-26, the protocol of that test section 5, the method page): Classical with an NVIDIA GPU:
# Leela with the history network and Stockfish complements (L-Leela); everywhere else (Rapid, Blitz, or no NVIDIA GPU): the Stockfish calculator (L-SF).
# Without Stockfish or its networks, Auto falls back to Leela with the network v1 (the previous default).
def _nvidia_ok():
    """An NVIDIA GPU that answers: nvidia-smi present AND listing a GPU (a driver installed but not loaded must not send Toby to Leela).
    Through toby2.nvidia_smi: one call per process, shared with Toby 2's GPU probe, output to a file and a hard wall-clock limit with the
    whole process tree killed (a hanging nvidia-smi or .bat wrapper cannot block the engine). `nvidia-smi -L` only when the card query
    itself is refused (a driver too old for compute_cap)."""
    try:
        import toby2 as _t2
    except Exception:  # noqa: BLE001
        return False
    if _t2.nvidia_rows():
        return True
    rc, _txt, err = _t2.nvidia_smi()
    if err:   # missing, failed to start or did not answer in time: no second call
        return False
    rc, txt, err = _t2.nvidia_smi(("-L",))
    return not err and rc == 0 and "GPU" in txt


HAS_NVIDIA = os.environ.get("TOBY_GPU", "1" if _nvidia_ok() else "0") == "1"


LEELA_BROKEN = False   # set when lc0 fails during a search: Toby then uses Stockfish instead


NVIDIA_BACKENDS = ("cuda", "cudnn", "onnx-trt", "onnx-cuda")   # lc0 backends that need the NVIDIA driver


def leela_no_driver():
    """lc0 is set to an NVIDIA backend (cuda, ...) but no NVIDIA GPU answers: it could not run, so it is never started (a cuda lc0 without
    the driver may hang until the start watchdog). The CPU backends of Toby-light (toby_backend.txt = onnx-cpu, ...) are not concerned."""
    return not HAS_NVIDIA and lc0_backend().startswith(NVIDIA_BACKENDS)


def leela_found():
    """lc0 (next to Toby, a full path, or on the PATH) and its network are there, lc0 can run here, and it has not failed on this machine."""
    return not LEELA_BROKEN and bool(os.path.exists(LC0) or shutil.which(LC0)) and os.path.exists(NET) and not leela_no_driver()


def sf_ready():
    return CORRECTOR_SF is not None and CORRECTOR_HSF is not None and os.path.exists(SF_PATH)


def calculator():
    """Calculator actually used for the next search: the option, or the Auto rule."""
    c = opts["calculator"]
    if c != "Auto":
        return c
    if not sf_ready():
        return "Leela"
    leela_ok = leela_found()   # Toby-light (no lc0): Stockfish even with an NVIDIA GPU
    return "Leela" if (opts["mode"] == "Classical" and HAS_NVIDIA and leela_ok and lc0_on_gpu()) else "Stockfish"

SF_EVAL_MOVES = 12   # Stockfish calculator: Toby's win/draw/loss is computed for the 12 most probable moves (Maia's value head on a CPU: ~35 ms per move)
SF_TRAIN_NODES = 300000   # budget of the training data of the current Stockfish network (Stockfish 19, 300k nodes); see sf_nodes_auto
sf_engine = None
try:   # Stockfish Q -> Leela Q table (nn/fit_sf_to_leela.py): Toby's win / draw / loss model was calibrated on Leela's values
    import json as _json
    _S2L = _json.loads((HERE / "nn" / "sf_to_leela.json").read_text(encoding="utf-8"))
except Exception:  # noqa: BLE001
    _S2L = None


try:   # win / draw / loss constants refitted on Stockfish's own values (nn/fit_wdl_sf.py), single-rating use; key "Classical|otb"
    _WDL_SF = _json.loads((HERE / "nn" / "wdl_sf_constants.json").read_text(encoding="utf-8"))["single"]
except Exception:  # noqa: BLE001
    _WDL_SF = {}


def wdl_sf_consts():
    """Stockfish constants for the current time control and venue (single-rating use), or None (then the Leela constants with the value table)."""
    c = _WDL_SF.get(f"{opts['mode']}|{venue()}") or _WDL_SF.get(f"{opts['mode']}|otb")
    return tuple(c) if c else None


def sf_to_leela_q(q):
    if not _S2L:
        return q
    xs, ys = _S2L["x_sf"], _S2L["y_leela"]
    if q <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if q <= xs[i]:
            t = (q - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] > xs[i - 1] else 0.0
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def sf_threads():
    """Option Stockfish Threads, else 1 thread in every time control (the networks are trained on single-thread values; 4 threads cost 0.014
    of log-likelihood in Classical on the dev set; v6, 4 November: multi-thread values are not reproducible, so not in Rapid / Blitz either)."""
    if opts["sf_threads"]:
        return opts["sf_threads"]
    return 1   # every time control: the networks were trained on single-thread values, and a multi-thread search is not reproducible


def sf_nodes_auto(E):
    """Node budget per position when the option Stockfish Nodes is 0. For now the budget of the network's training data;
    to be replaced by a continuous function of the strength and the time control fitted on the node-budget study (the development log)."""
    return SF_TRAIN_NODES


def sf_cache_key(b, nodes, threads, sf_name):
    """The position, the last 8 moves (repetitions), the node budget, the threads and the Stockfish version: what evaluate() answers for."""
    return "|".join([b.fen(), " ".join(m.uci() for m in b.move_stack[-8:]), str(nodes), str(threads), sf_name, "v3"])   # v3: extension keys keep the moves' order (2026-11-12)


class SFCache:
    """Bounded cache of Stockfish answers, least recently USED dropped first (a position read again, e.g. a common opening, moves to the front).
    TOBY_SF_CACHE_SIZE entries (default 20,000; 0 turns it off). With TOBY_SF_CACHE_FILE (a path) it is also kept on disk: one compact JSON line
    per answer, appended; rewritten with the current entries, most recently used last, when it grows past twice the size. The file is loaded in
    a background thread, so it never delays the start: until it is loaded, searches simply miss the cache. In memory each entry is kept as
    its JSON text (about 1 KB: every move's value, the lines of the 8 best moves, 8 half-moves each), decoded when read."""

    def __init__(self):
        import json
        from collections import OrderedDict
        self.json = json
        self.size = int(os.environ.get("TOBY_SF_CACHE_SIZE", "20000"))
        self.path = os.environ.get("TOBY_SF_CACHE_FILE") or None
        self.d = OrderedDict()
        self.lock = threading.Lock()
        self.lines = 0
        if self.path and self.size > 0 and os.path.exists(self.path):
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            ft_loaded.wait(300)   # after Maia (and the fine-tuned Maia) are loaded: the start is never slowed down
            loaded = []
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    k = line[7:line.index('", ', 7)] if line.startswith('{"k": "') else None   # the key, without decoding the line
                    if k is not None:
                        loaded.append((k, line.rstrip("\n")))
            loaded = loaded[-self.size:]
            with self.lock:
                for k, txt in reversed(loaded):   # older than this session's entries, most recent of the file first
                    if k not in self.d:
                        self.d[k] = txt
                        self.d.move_to_end(k, last=False)
                while len(self.d) > self.size:
                    self.d.popitem(last=False)
                self.lines += len(loaded)
        except Exception:  # noqa: BLE001
            pass

    def get(self, k):
        if self.size <= 0:
            return None
        with self.lock:
            txt = self.d.get(k)
            if txt is None:
                return None
            self.d.move_to_end(k)
        try:
            r = self.json.loads(txt)
            return r["v"], r["m"]
        except Exception:  # noqa: BLE001
            return None

    def put(self, k, v):
        if self.size <= 0:
            return
        vals, meta = v
        top = sorted(vals, key=lambda u: -vals[u])[:8]
        meta = {"depth": meta.get("depth"), "nodes": meta.get("nodes"), "mate": {u: m for u, m in (meta.get("mate") or {}).items() if m is not None},
                "pv": {u: " ".join(meta.get("pv", {}).get(u, u).split()[:8]) for u in top}}
        txt = self.json.dumps({"k": k, "v": vals, "m": meta}, separators=(",", ":")).replace('{"k":"', '{"k": "', 1).replace('","v":', '", "v":', 1)
        with self.lock:
            self.d[k] = txt
            self.d.move_to_end(k)
            while len(self.d) > self.size:
                self.d.popitem(last=False)
            if not self.path:
                return
            try:
                if self.lines >= 2 * self.size:   # compaction: the current entries, most recently used last
                    tmp = self.path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        for t in self.d.values():
                            fh.write(t + "\n")
                    os.replace(tmp, self.path)
                    self.lines = len(self.d)
                else:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(txt + "\n")
                    self.lines += 1
            except Exception:  # noqa: BLE001
                pass


SF_CACHE = SFCache()


class StockfishCalc:
    """Stockfish through python-chess: every legal move evaluated in one MultiPV search."""

    def __init__(self, fixed_threads=None):
        """fixed_threads: an engine of the review pool or the second engine (1 thread whatever the options). Every search clears the hash
        first: the answer depends on the position only, not on which engine got it or what it searched before."""
        import chess.engine
        kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
        kw["stderr"] = subprocess.DEVNULL   # never the GUI's stderr handle (may be invalid: a GUI-subsystem parent without redirect)
        self.eng = chess.engine.SimpleEngine.popen_uci(SF_PATH, **kw)
        try:
            match_priority(self.eng.protocol.transport.get_pid())
        except Exception:  # noqa: BLE001
            pass
        check_version("Stockfish", self.eng.id.get("name", ""), TESTED_SF)
        self.fixed = fixed_threads
        self.threads = None
        self.last = None
        self.configure()

    def configure(self):
        t = self.fixed or sf_threads()
        if t != self.threads:
            self.eng.configure({"Threads": t, "Hash": 64 * t})
            self.threads = t

    def evaluate(self, b, nodes, on_depth=None):
        """{uci: centipawns for the side to move (mates +-10000)} of every legal move, from the deepest iteration COMPLETED within the node budget
        (the training data are read the same way: nn/dataset.at_budget); None if stopped before the end. The details of the last call
        (depth, nodes, mate distances, principal variations) are in self.last. Answers are cached (SF_CACHE: the position, the last 8 moves,
        the node budget, the threads; also on disk with TOBY_SF_CACHE_FILE). on_depth(d, {uci: cp}): called during the search each time an
        iteration of depth >= 8 has a value for every legal move (pass-1 streaming, Toby 2's speculative complements); the result is unchanged."""
        import chess.engine
        self.configure()
        key = sf_cache_key(b, nodes, self.threads, self.eng.id.get("name", ""))
        hit = SF_CACHE.get(key)
        if hit is not None:
            self.last = hit[1]
            return dict(hit[0])
        n = b.legal_moves.count()
        self.eng.configure({"Clear Hash": None})   # the answer depends on the position only, not on what this engine searched before (same values one by one, in a batch, in any order)
        per, nod, last, mates, pvs = {}, {}, {}, {}, {}
        fb = chess.Board(b.fen())   # the position alone (FEN), as the training data (nn/arbiter.py): no move history
        with self.eng.analysis(fb, chess.engine.Limit(nodes=nodes), multipv=n) as an:
            for info in an:
                if stop_flag.is_set():
                    an.stop()
                    return None
                if "pv" in info and info["pv"] and "score" in info:
                    sc = info["score"].pov(b.turn)
                    cp = (10000 if sc.mate() > 0 else -10000) if sc.is_mate() else sc.score()
                    u = info["pv"][0].uci()
                    last[u] = cp
                    d = info.get("depth", 0)
                    if d >= 8 and not info.get("lowerbound") and not info.get("upperbound"):   # checkpoints from depth 8 on, as nn/arbiter.py
                        per.setdefault(d, {})[u] = cp
                        nod[d] = max(nod.get(d, 0), info.get("nodes", 0))
                        mates.setdefault(d, {})[u] = sc.mate() if sc.is_mate() else None
                        pvs.setdefault(d, {})[u] = " ".join(m.uci() for m in info["pv"][:12])
                        if on_depth is not None and len(per[d]) == n:
                            try:
                                on_depth(d, dict(per[d]))
                            except Exception:  # noqa: BLE001
                                pass
        full = [d for d, v in per.items() if len(v) == n]   # depths where every legal move has a value (nn/dataset.at_budget)
        done = [d for d in full if nod[d] <= nodes]
        if full:   # the deepest completed within the budget, else the shallowest completed one
            d = max(done) if done else min(full)
            res, self.last = per[d], {"depth": d, "nodes": nod[d], "mate": mates[d], "pv": pvs[d]}
        elif len(last) == n:
            d = max(per) if per else 0
            res, self.last = last, {"depth": d, "nodes": nod.get(d, 0), "mate": mates.get(d, {}), "pv": pvs.get(d, {})}
        else:
            return None
        SF_CACHE.put(key, (dict(res), self.last))
        return dict(res)

    def extend(self, b, moves, nodes, cancel=None):
        """EXPERIMENTAL (V2): {uci: cp} of a search on the given moves only (MultiPV over them, hash cleared, final values), as
        nn/sf_adaptive/pilot.search. Cached like evaluate (the moves, in their order, are part of the key)."""
        import chess.engine
        self.configure()
        # the moves IN THEIR ORDER are part of the key: Stockfish's MultiPV search over root moves depends on their order (2026-11-12: the same
        # three moves in another order gave up to 25 cp apart), and the data searched each candidate list in its own order (nn/sf_adaptive/extend.py)
        key = sf_cache_key(b, f"ext{nodes}:{','.join(moves)}", self.threads, self.eng.id.get("name", ""))
        hit = SF_CACHE.get(key)
        if hit is not None:
            return dict(hit[0])
        if stop_flag.is_set():
            return None
        self.eng.configure({"Clear Hash": None})
        mv = [chess.Move.from_uci(m) for m in moves]
        if cancel is None:
            infos = self.eng.analyse(chess.Board(b.fen()), chess.engine.Limit(nodes=nodes), multipv=len(mv), root_moves=mv, game=object())   # FEN only, as the data
        else:   # cancellable (Toby 2's speculative complements): the same search; analyse() is analysis() + wait() + multipv
            with self.eng.analysis(chess.Board(b.fen()), chess.engine.Limit(nodes=nodes), multipv=len(mv), root_moves=mv, game=object()) as an:
                for _ in an:
                    if cancel.is_set():
                        an.stop()
                        return None
                infos = an.multipv
            if cancel.is_set():
                return None
        res = {}
        for i in infos:
            if "pv" in i and i["pv"] and "score" in i:
                sc = i["score"].pov(b.turn)
                res[i["pv"][0].uci()] = (10000 if sc.mate() > 0 else -10000) if sc.is_mate() else sc.score()
        if stop_flag.is_set():
            return None
        SF_CACHE.put(key, (dict(res), {}))
        return dict(res)

    def quit(self):
        try:
            self.eng.quit()
        except Exception:  # noqa: BLE001
            try:
                self.eng.close()
            except Exception:  # noqa: BLE001
                pass


def stop_stockfish():
    global sf_engine
    if sf_engine is not None:
        sf_engine.quit()
        sf_engine = None


atexit.register(stop_stockfish)


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


# ----------------------------------------------------------------------------------------------------- neural corrector
# A small network trained to correct Maia's probabilities with Leela's evaluations (nn/corrector.py, trained by nn/train.py). When its weights are present
# (nn/corrector_weights.json, or the file named by TOBY_NN) it replaces the rule above for the final distribution (option Move Model = Network);
# Move Model = Rule keeps the hand-made rule. The quick provisional display (static values) always uses the rule.
NN_PATH = os.environ.get("TOBY_NN", str(HERE / "nn" / "corrector_weights.json"))
CORRECTOR = None
_NNC = None
try:
    if os.path.exists(NN_PATH):
        sys.path.insert(0, str(HERE / "nn"))
        sys.path.insert(0, str(HERE))
        import corrector as _NNC  # noqa: E402   (nn/corrector.py in the repository; frozen into toby.exe in the package)
        CORRECTOR = _NNC.Corrector(NN_PATH)
except Exception:  # noqa: BLE001
    CORRECTOR = None
CORRECTOR_SF = None
try:
    if _NNC is not None and os.path.exists(SF_NN_PATH):
        CORRECTOR_SF = _NNC.Corrector(SF_NN_PATH)
except Exception:  # noqa: BLE001
    CORRECTOR_SF = None
# History network (sealed test v2: L-Leela; Move Model = History, or Auto with the Auto calculator choosing Leela): Maia given the previous positions; two weight files, one for the
# 64-node Leela complements and one for complements taken from Stockfish (same quality on the dev set, much faster: one Stockfish search on the
# CPU while Leela searches on the GPU, instead of one small Leela search per move).
CORRECTOR_H = CORRECTOR_HSF = None
try:
    if _NNC is not None:
        if os.path.exists(HERE / "nn" / "corrector_hist_weights.json"):
            CORRECTOR_H = _NNC.Corrector(str(HERE / "nn" / "corrector_hist_weights.json"))
        if os.path.exists(HERE / "nn" / "corrector_hist_sf_weights.json"):
            CORRECTOR_HSF = _NNC.Corrector(str(HERE / "nn" / "corrector_hist_sf_weights.json"))
except Exception:  # noqa: BLE001
    CORRECTOR_H = CORRECTOR_HSF = None
# Networks retrained with the fine-tuned Maia in Classical (the method page); used only with it (see ft_on).
CORRECTOR_SF_FT = CORRECTOR_HSF_FT = CORRECTOR_HYB = CORRECTOR_HYB_ON = CORRECTOR_ACC_MIX = CORRECTOR_HYB_MIX = None
try:   # the release's networks (nn/release_v6.json); an environment variable (a candidate under test) overrides each entry
    with open(HERE / "nn" / "release_v6.json", encoding="utf-8") as _rf:
        RELEASE_CFG = json.load(_rf)
except Exception:  # noqa: BLE001
    RELEASE_CFG = {}
HYB_MIX_W = float(os.environ.get("TOBY_HYB_MIX_W", RELEASE_CFG.get("hyb_mix_w", 0.5)))   # weight of the second classical hybrid network (HF: F1g, 0.5)
ACC_MIX_W = float(os.environ.get("TOBY_ACC_MIX_W", RELEASE_CFG.get("acc_mix_w", 0.3)))   # weight of the Accurate network in the Accurate ensemble


def _nn_file(env, name, key=None):
    """Network file: the environment variable (a candidate under test), else the release's choice (nn/release_v6.json), else nn/<name>."""
    if os.environ.get(env):
        return os.environ[env]
    return str(HERE / "nn" / (RELEASE_CFG.get(key) or name)) if key else str(HERE / "nn" / name)


def _mix_file(env, key):
    """Second network of an ensemble: the environment variable (empty = no ensemble), else the release's choice, else none."""
    if env in os.environ:
        return os.environ[env] or None
    return str(HERE / "nn" / RELEASE_CFG[key]) if RELEASE_CFG.get(key) else None


try:
    if _NNC is not None:
        # sealed test v5 (the method page): Fast = T5, Accurate = T10 (V2 Stockfish values as extra inputs), Hybrid = H8
        if os.path.exists(HERE / "nn" / "corrector_sf_ft_level2_weights.json"):
            CORRECTOR_SF_FT = _NNC.Corrector(str(HERE / "nn" / "corrector_sf_ft_level2_weights.json"))
        if os.path.exists(_nn_file("TOBY_NN_ACC", "corrector_hist_sf_ft_level2_sfa_weights.json", "acc")):
            CORRECTOR_HSF_FT = _NNC.Corrector(_nn_file("TOBY_NN_ACC", "corrector_hist_sf_ft_level2_sfa_weights.json", "acc"))
        if os.path.exists(_nn_file("TOBY_NN_HYB", "corrector_sf_ft_level2_sfa_pol_weights.json", "hyb")):
            CORRECTOR_HYB = _NNC.Corrector(_nn_file("TOBY_NN_HYB", "corrector_sf_ft_level2_sfa_pol_weights.json", "hyb"))
        # Fast in Rapid / Blitz (sealed test v6 candidates HO6a/b): Stockfish 300k over every move (no V2) + Leela's raw policy, released Maia
        # Accurate as an ensemble (sealed test v6 candidate AX): the Accurate network ^ w times the classical hybrid network ^ (1 - w)
        # Fast as an ensemble (candidate HF): the classical hybrid network ^ (1 - w) times a second one (F1g) ^ w, same inputs
        _m = _mix_file("TOBY_NN_HYB_MIX", "hyb_mix")
        if _m and os.path.exists(_m):
            CORRECTOR_HYB_MIX = _NNC.Corrector(_m)
        _m = _mix_file("TOBY_NN_ACC_MIX", "acc_mix")
        if _m and os.path.exists(_m):
            CORRECTOR_ACC_MIX = _NNC.Corrector(_m)
        if os.path.exists(_nn_file("TOBY_NN_HYB_ONLINE", "corrector_hyb_online_weights.json", "hyb_online")):
            CORRECTOR_HYB_ON = _NNC.Corrector(_nn_file("TOBY_NN_HYB_ONLINE", "corrector_hyb_online_weights.json", "hyb_online"))
except Exception:  # noqa: BLE001
    CORRECTOR_SF_FT = CORRECTOR_HSF_FT = CORRECTOR_HYB = CORRECTOR_HYB_ON = CORRECTOR_ACC_MIX = CORRECTOR_HYB_MIX = None
# V2 Stockfish values (sealed test v5, amendments 2-4): 150k nodes over every move, then 250k on the candidates (fine-tuned Maia's top 3 +
# Stockfish's top 2). Used by Precision Accurate (network T10) and Hybrid (H8, with Leela's raw policy), Classical with the fine-tuned Maia.
V2_BASE, V2_EXT = 150000, 250000
_EXP_LC0_LOCK = threading.Lock()


PRECISIONS = ["Accurate", "Fast"]   # Fast = the hybrid path (Stockfish + Leela's raw policy) where it exists; "Hybrid" is accepted as an alias of Fast
_FALLBACK_SAID = [False]


def hybrid_wanted():
    """The hybrid path is wanted: Precision Fast, or Accurate when its Auto rule gives Stockfish (no NVIDIA GPU, or Leela missing), in Classical
    with the fine-tuned Maia and network H8 present. Leela runs on the GPU or on the CPU."""
    if not (bool(opts.get("hybrid")) or opts["calculator"] == "Auto") or calculator() != "Stockfish":
        return False
    if opts["mode"] == "Classical":
        return CORRECTOR_HYB is not None and ft_on()
    return CORRECTOR_HYB_ON is not None


def hybrid_net():
    """Network of the hybrid path for the current time control."""
    return CORRECTOR_HYB if opts["mode"] == "Classical" else CORRECTOR_HYB_ON


def hybrid_on():
    """The hybrid path is wanted and Leela (lc0 + its network) is there and has not failed on this machine."""
    return hybrid_wanted() and leela_found()


def say_fallback(reason):
    """Leela missing or broken while the hybrid path is wanted: said once, then Fast runs on Stockfish alone (the Stockfish-only network of the time control)."""
    if not _FALLBACK_SAID[0]:
        _FALLBACK_SAID[0] = True
        out(f"info string Leela unavailable ({reason}): Fast runs on Stockfish alone")


def v2_candidates(pm, base):
    """Candidates of the extension (as nn/sf_adaptive/pilot.triggers): Maia's top 3 + Stockfish's top 2 at the base budget, legal, no duplicate."""
    c = list(dict.fromkeys([m for m, _ in sorted(pm.items(), key=lambda kv: -kv[1])[:3]] + sorted(base, key=base.get, reverse=True)[:2]))
    return [m for m in c if m in base]


def n_cores():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        return os.cpu_count() or 1


def _cgroup_cpu_quota():
    """The container's CPU quota in cores (cgroup v2 cpu.max, or v1 cpu.cfs_quota_us / cpu.cfs_period_us), or None when there is none."""
    paths = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0" and parts[1] == "":   # cgroup v2: "0::/path"
                paths.append(Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "cpu.max")
    except Exception:  # noqa: BLE001
        pass
    paths.append(Path("/sys/fs/cgroup/cpu.max"))
    for f in paths:
        try:
            q, p = f.read_text().split()[:2]
            if q != "max" and int(p) > 0:
                return int(q) / int(p)
            if q == "max":
                return None
        except Exception:  # noqa: BLE001
            continue
    for d in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        try:
            q = int(Path(d, "cpu.cfs_quota_us").read_text())
            p = int(Path(d, "cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                return q / p
            return None
        except Exception:  # noqa: BLE001
            continue
    return None


_USABLE = []


def usable_cores():
    """Cores this process can really use (cached): the CPUs it may run on (sched_getaffinity), capped by the container's CPU quota if
    there is one (RunPod's A40 pods show 96 CPUs with a quota of 7.65 cores). Used only to decide the optional parallel work below;
    n_cores() (affinity only) still decides the thread counts, which must not change."""
    if not _USABLE:
        n = float(n_cores())
        q = _cgroup_cpu_quota()
        _USABLE.append(min(n, q) if q else n)
    return _USABLE[0]


def t2_cores_used():
    """Cores a Toby 2 search already keeps busy: B's torch threads, Stockfish pass 1 and the second Stockfish, and lc0 for the Leela head."""
    th = opts["t2_threads"] or (2 if n_cores() >= 4 else 1)
    cfg = T2.get("cfg") or {}
    return th + 2 + (1 if cfg.get("head") else 0)


_SPEC = {"eng": None, "busy": threading.Lock()}


def spec_engine():
    """Second single-thread Stockfish (2 cores or more): the V2 extension on Maia's top 3 starts with the base search, and the extensions of
    a rating grid run two at a time. Same values: every extension clears the hash. None on one core."""
    if n_cores() < 2 or not os.path.exists(SF_PATH):
        return None
    if _SPEC["eng"] is None:
        try:
            _SPEC["eng"] = StockfishCalc(fixed_threads=1)
        except Exception:  # noqa: BLE001
            return None
    return _SPEC["eng"]


def maia_top3(pm, b):
    """The Maia part of the V2 candidates (as v2_candidates), known before the base search."""
    legal = {m.uci() for m in b.legal_moves}
    return [m for m, _ in sorted(pm.items(), key=lambda kv: -kv[1])[:3] if m in legal]


def sf_v2(b, pm, base=None, eng=None, spec=None, pre=None, info=None):
    """V2 values {uci: cp}: the base search over every move, the candidates' values replaced by a 250k search on them (hash cleared), as
    nn/sf_adaptive/extend.py computed the training values. None if stopped.
    spec: a second engine: while eng runs the base search, spec runs the extension on Maia's top 3; used when Stockfish's top 2 are among
    them (the candidates are then exactly Maia's top 3, so the same search), else the extension runs after as usual. pre: extensions already
    computed, {tuple(candidates): values}. info: a dict that receives "v2info" = (best move of the base search, best move after the
    extension, {uci: (base cp, extension cp)} of the extended candidates), an input of the section-10 networks (nn/corrector.build_inputs)."""
    eng = eng or sf_engine
    guess, sres, th = None, {}, None
    if base is None and spec is not None and _SPEC["busy"].acquire(blocking=False):
        guess = maia_top3(pm, b)

        def _run():
            try:
                r = spec.extend(b, guess, V2_EXT) if len(guess) >= 2 else None
                if r is not None:
                    sres["v"] = r
            except Exception:  # noqa: BLE001
                pass
            finally:
                _SPEC["busy"].release()
        th = WThread(target=_run, daemon=True)
        th.start()
    if base is None:
        base = eng.evaluate(b, V2_BASE)
    if base is None:
        return None
    cands = v2_candidates(pm, base)
    b0 = max(base, key=base.get) if base else None   # first maximum in Stockfish's MultiPV order, as the data
    if len(cands) < 2:
        if info is not None:
            info["v2info"] = (b0, b0, {})
        return dict(base)
    ext = None
    if pre is not None and tuple(cands) in pre:
        ext = pre[tuple(cands)]
    elif th is not None and cands == guess:
        th.join()
        ext = sres.get("v")
    if ext is None:
        ext = eng.extend(b, cands, V2_EXT)
    if ext is None:
        return None
    v = dict(base)
    v.update({u: cp for u, cp in ext.items() if u in v})
    if info is not None:
        info["v2info"] = (b0, max(v, key=v.get), {u: (base[u], cp) for u, cp in ext.items() if u in base})
    return v


def preload_raw_policy():
    """Precision Hybrid selected: start lc0 now; if it fails, Hybrid searches as Fast (LEELA_BROKEN), never waits."""
    global LEELA_BROKEN
    try:
        t = time.time()
        leela_raw_policy(chess.Board())
        out(f"info string toby timing leela ready {time.time() - t:.2f}s (lc0 start, network load, first evaluation)")
    except Exception:  # noqa: BLE001
        LEELA_BROKEN = True
        stop_all_lc0()


def leela_raw_policy(b):
    """Leela's raw policy {uci: prior} of the position: one network evaluation (1 node), as the features of the hybrid's training data."""
    global lc0, LEELA_BROKEN
    with _EXP_LC0_LOCK:   # the preload of set_option and a search must not start two lc0
        if LEELA_BROKEN:   # a start that failed while this call waited for the lock (e.g. the preload): never start lc0 a second time
            raise RuntimeError("lc0 failed on this machine")
        try:
            ensure_lc0(light=calculator() != "Leela")   # Fast: one-node evaluations only (the light lc0); a Leela search needs the full one
            lc0.search(b.fen(), 1)
        except Exception:
            LEELA_BROKEN = True   # marked before the lock is released, so that a search waiting for it fails at once
            raise
        return {u: round(float(v), 5) for u, v in dict(getattr(lc0, "last_policy", {})).items()}   # 5 decimals, as the data (nn/retrain_v6/raw_policy.py)


MOVE_MODELS = ["Auto", "Network", "Rule"] + (["History"] if CORRECTOR_H is not None else [])
# Default move model per time control: the network only where it passed its sealed test (the protocol of that test: N1 and N2), the rule elsewhere.
# "Auto" follows this table; the option Move Model = Network / Rule forces one model in every time control.
NETWORK_DEFAULT = {"Classical", "Rapid", "Blitz"}   # sealed test of 2026-10-20: N1 and N2 pass in the three (the method page)
opts["move_model"] = os.environ.get("TOBY_MOVE_MODEL", "Auto") if CORRECTOR else "Rule"


def own_maia_rating(b):
    """Maia rating of the player's own rating (second Maia input of the network): FIDE and Lichess ratings as entered, Chess.com through the Lichess equivalent."""
    wr, br = opts["white_rating"], opts["black_rating"]
    r = (wr or br or DEFAULT_RATING) if b.turn == chess.WHITE else (br or wr or DEFAULT_RATING)
    if ft_on():   # the networks of the fine-tuned Maia were trained with this second input equal to the first (E + 700)
        return rule_params(strength_E(r, opts["source"]))[0]
    if opts["source"].startswith("Chess.com"):
        r = from_fide(strength_E(r, opts["source"]), "lichess", opts["mode"].lower() if opts["mode"] != "Classical" else "classical")
    return int(min(max(r, 600), 3500))


def network_on():
    if CORRECTOR is None:
        return False
    if opts["move_model"] == "Auto" and auto_hist():
        return False
    if opts["move_model"] == "Auto":
        return opts["mode"] in NETWORK_DEFAULT
    return opts["move_model"] == "Network"


def auto_hist():
    """Move Model = Auto with the Auto calculator choosing Leela: the history network with Stockfish complements (L-Leela)."""
    return opts["calculator"] == "Auto" and sf_ready() and calculator() == "Leela" and CORRECTOR_H is not None


def acc_mix(d, b, pm, E, x, opp, sfraw, v2info, pol1):
    """Ensemble AX: d (the Accurate network's distribution) ^ ACC_MIX_W times the hybrid network's ^ (1 - ACC_MIX_W), renormalised over the
    legal moves. The hybrid network gets exactly the inputs of the Fast path: the V2 values, their search details, Leela's 1-node policy."""
    if CORRECTOR_ACC_MIX is None or not sfraw or pol1 is None or not d:
        return d
    lee_s, root_s = _NNC.sf_pseudo_leela(sfraw)
    c = CORRECTOR_ACC_MIX
    moves, ctx, mv, base = _NNC.build_inputs(b, _r6(pm), lee_s, pol1, {}, root_s, x, E, opts["mode"].lower(), venue(),
                                             E_opp=(opp[0] if opp else None), pm2=None, history=True, sf_budget=budget_input(c, V2_BASE),
                                             v2info=v2info)
    dh = c.distribution(ctx, mv, base, moves)
    w = ACC_MIX_W
    z = {u: math.exp(w * math.log(max(d.get(u, 0.0), 1e-12)) + (1 - w) * math.log(max(dh.get(u, 0.0), 1e-12))) for u in d}
    t = sum(z.values())
    return {u: v / t for u, v in z.items()}


def hist_on():
    if CORRECTOR_H is None:
        return False
    return opts["move_model"] == "History" or (opts["move_model"] == "Auto" and auto_hist())


def move_distribution(b, pm, lee, extra, E, x, opp, root, pol, pm2=None, cor=None, sf_shallow=None, v2info=None):
    """Final move distribution: the network when it is on (same inputs as in training, nn/corrector.build_inputs), else the hand-made rule.
    cor: a history network (Move Model = History), which reads Maia with the previous positions. Returns ({uci: probability}, {uci: loss} (rule only), Leela's best move)."""
    if not lee or (cor is None and not network_on()):
        return toby_distribution(pm, lee, extra, E)
    c = cor or CORRECTOR
    moves, ctx, mv, base = _NNC.build_inputs(b, _r6(pm), lee, {u: round(v, 5) for u, v in (pol or {}).items()}, extra, root, x, E,
                                             opts["mode"].lower(), venue(), E_opp=(opp[0] if opp else None),
                                             pm2=_r6(pm2) if c.meta.get("use_pm2") else None, history=cor is not None,
                                             sf_shallow=sf_shallow, v2info=v2info)   # sf_shallow / v2info: Accurate network only (V2 values and V2 search details)
    best = max(lee, key=lambda u: (lee[u][0], lee[u][3]))
    return c.distribution(ctx, mv, base, moves), {}, best


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


def wdl_model(es_leela, maia_ldw, E, D=None, consts=None):
    """(win, draw, loss) of the side TO MOVE in a position: es_leela = expected score Leela gives that side; maia_ldw = Maia's [loss, draw, win] for that side;
    E = strength of that side; D = its strength minus its opponent's, in Elo (None: opponent not given -> the single-rating constants)."""
    c = consts or (WDL_SINGLE if D is None else WDL_OPPONENT)[opts["mode"]][venue()]
    wl, wm, a2, b2, d2, wu, ke, wd = c[:8]
    l, d, w = maia_ldw
    eL, eM, md = _logit(es_leela), _logit(w + d / 2), _logit(d)
    t = (wl + wu * max(0.0, (E - 2200) / 1000)) * eL + wm * eM + ke * eL * abs(eL) + (wd * D / 1000 if D is not None else 0.0)
    lnu = a2 + b2 * abs(eL) + d2 * md
    if len(c) > 8:   # extended base of an Endgame Evaluation candidate (amendment 8): a draw level depending on the strength
        z = (E - 2000) / 1000
        lnu += c[8] * z + c[9] * z * z
    nu = math.exp(lnu)
    wv, lv = math.exp(t / 2), math.exp(-t / 2)
    z = wv + lv + nu
    return (wv / z, nu / z, lv / z)


def eval_elo(E, x):
    """Rating given to Maia's value head: E clamped to [1000, 2700] in Classical, the Maia input rating (at most 2700) otherwise."""
    return int(max(1000, min(VALUE_ELO_CAP, round(E)))) if opts["mode"] == "Classical" else int(min(VALUE_ELO_CAP, x))


_MV = {}


def _maia_values_plan(boards, elo, oppo=None, allpol=False):
    """What maia_values_batch computes now: (todo = the boards not cached, in their order; pol; need) for the current caches and options."""
    we = wdl_endgame(None)
    pol = we is not None and we[1] == "valpass"
    need = (lambda c: allpol or _WE.applies(c.fen())) if pol else (lambda c: False)   # allpol: candidate M's Q reads every position's stats
    todo = [c for c in boards if (c.fen(), elo, oppo) not in _MV or (need(c) and (c.fen(), elo, oppo) not in _MVP)]
    return todo, pol, need


def _maia_values_key(todo, elo, oppo, pol, need):
    """Identity of one Maia value pass: every board in full (root position and moves, in the batch order), both ratings and what is kept."""
    return (tuple((c.root().fen(), tuple(m.uci() for m in c.move_stack)) for c in todo), elo, oppo, pol, tuple(bool(need(c)) for c in todo))


def _maia_values_run(todo, elo, oppo, pol, need):
    """The network pass of maia_values_batch on exactly `todo` (one batch): (values [l, d, w] per board, {key: policy stats})."""
    eng, torch, _ = maia
    cfg = eng.cfg
    mvp = {}
    with MAIA_LOCK, torch.no_grad():
        toks = []
        for c in todo:
            eng.board = c
            eng._reset_history()
            toks.append(eng._tokens_from_history(eng.history))
        se = torch.full((len(todo),), elo, dtype=torch.long, device=cfg.device)
        so = se if oppo is None else torch.full((len(todo),), oppo, dtype=torch.long, device=cfg.device)
        lm, v, _ = eng.model(torch.stack(toks).to(cfg.device), se, so)
        vals = torch.softmax(v.float(), dim=-1).tolist()
        if pol:
            for i, c in enumerate(todo):
                if need(c):
                    mvp[(c.fen(), elo, oppo)] = _valpass_policy(lm[i], c)
    return vals, mvp


def maia_values_batch(boards, elo, oppo=None, allpol=False):
    """Maia's value of several positions in one pass of the network (the engine's evaluation of every shown move); results go to the cache.
    With the option Endgame Evaluation, the policy stats of the same pass are kept too (positions where the correction applies).
    A pass of exactly the same batch already made by the Move Evaluations prefetch (EvalPrefetch) is used instead of a new one: same
    boards in the same order, same ratings, so the same numbers."""
    todo, pol, need = _maia_values_plan(boards, elo, oppo, allpol)
    if not todo or maia is None:
        return
    pre = _EVPRE[0]
    got = pre.take_batch(_maia_values_key(todo, elo, oppo, pol, need)) if pre is not None else None
    vals, mvp = got if got is not None else _maia_values_run(todo, elo, oppo, pol, need)
    if pol:
        if len(_MVP) > 4000:
            _MVP.clear()
        _MVP.update(mvp)
    if len(_MV) > 4000:
        _MV.clear()
    for c, w in zip(todo, vals):
        _MV[(c.fen(), elo, oppo)] = w


def maia_value_cached(board, elo, oppo=None):
    k = (board.fen(), elo, oppo)
    if k not in _MV:
        if len(_MV) > 4000:
            _MV.clear()
        pre = _EVPRE[0]
        got = pre.take_single(board, elo, oppo) if pre is not None else None   # the same single pass, made by the prefetch
        _MV[k] = got if got is not None else maia_value(board, elo, oppo)
    return _MV[k]


def _board_id(b):
    return (b.root().fen(), tuple(m.uci() for m in b.move_stack))


class EvalPrefetch:
    """Toby 2's Move Evaluations prefetch: Maia's value passes of the final stage, made while the Stockfish complements run.
    When Stockfish pass 1 is done, B's heads on the pass-1 values give a provisional distribution; its shown moves give the batch that
    toby_evals would send to maia_values_batch, and that batch is computed now, followed by the position's own single value pass.
    The final stage uses a prefetched pass ONLY when it would make exactly the same pass: the same boards (root position and moves) in
    the same order, with the same ratings and the same policy statistics kept (batch), or the same board and ratings (single). The
    network is deterministic for the same inputs, so the numbers are those the final stage would compute; any other case computes as
    before. Output-identical by construction; it changes only when the work is done."""

    def __init__(self):
        self.planned = threading.Event()   # the batch key is known (or the prefetch gave up)
        self.done = threading.Event()      # the batch pass is finished
        self.single_done = threading.Event()
        self.key = None                    # _maia_values_key of the prefetched batch
        self.batch = None
        self.single_key = None             # (board id, elo, oppo) of the prefetched single pass
        self.single = None                 # its [l, d, w]
        self.th = None
        self.stats = {"batch_hit": 0.0, "single_hit": 0.0}

    def start(self, fn):
        def _run():
            try:
                fn(self)
            except Exception:  # noqa: BLE001   (a failed prefetch only means the final stage computes everything itself)
                pass
            finally:
                self.planned.set()
                self.done.set()
                self.single_done.set()
        self.th = WThread(target=_run, daemon=True)
        self.th.start()

    def take_batch(self, key):
        if self.th is None:
            return None
        self.planned.wait(5.0)
        if self.key is None or key != self.key:
            return None
        self.done.wait(5.0)
        if self.batch is None or not self.done.is_set():
            return None
        self.stats["batch_hit"] = 1.0
        b, self.batch = self.batch, None
        return b

    def take_single(self, board, elo, oppo):
        if self.th is None:
            return None
        self.planned.wait(5.0)
        if self.single_key is None or (_board_id(board), elo, oppo) != self.single_key:
            return None
        self.single_done.wait(5.0)
        v, self.single = self.single, None
        if v is not None and self.single_done.is_set():
            self.stats["single_hit"] = 1.0
            return v
        return None


_EVPRE = [None]   # the current search's EvalPrefetch (Toby 2), or None


ENDGAME_CANDIDATES = ["P1a", "P1b", "P1c", "P1d", "P2a", "P2c", "M"]   # nn/wdl_ext_params_v4.json (protocol NN v6, amendments 6-6e, 8)
# M (nn/wdl_mix.py, Yosha's definition): the position's W/D/L = sum over the evaluated moves of Toby's probability x the move's W/D/L (an
# expectation never above its best move); each move's W/D/L = P2a's move view, + the terms Q (nn/wdl_mix_q_classical.json) in Classical,
# Fast, both ratings given.
MIX_Q_FILE = "wdl_mix_q_classical.json"
_MIXQ = []   # the terms Q, loaded at the first use
ENDGAME_PARAMS = "wdl_ext_params_v4.json"   # in nn/ next to Toby (read by path: the module may be frozen into toby.exe)
_WE = None   # nn/wdl_endgame_p1.py, loaded at the first use of the option Endgame Evaluation


def _endgame_cand():
    """The candidate whose parameters the option names: M uses P2a's (base and endgame terms)."""
    c = opts.get("wdl_endgame", "Off")
    return "P2a" if c == "M" else c


def mix_on():
    return opts.get("wdl_endgame", "Off") == "M"


def mix_q(src, opp):
    """The terms Q of candidate M (Classical, Fast, both ratings given), else None."""
    if not (mix_on() and src == "sf" and opp and opts["mode"] == "Classical") or wdl_endgame(None) is None:   # Q reads the endgame module's policy stats
        return None
    if not _MIXQ:
        try:
            _MIXQ.append(json.loads((HERE / "nn" / MIX_Q_FILE).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            _MIXQ.append(None)
    return _MIXQ[0]


def apply_mix_q(wdl, es, mv, top, ent, E, men, qp):
    """Q on a move's W/D/L (wdl = (w, d, l) of the side to move after the move, Davidson): t += odd terms, ln nu += even terms, from the
    position after the move only: its value es (expected score, Leela's scale), Maia's value mv = [l, d, w] and value-pass policy (top,
    entropy), the strength E of its side to move, its number of men (nn/wdl_mix.py)."""
    def lg(p):
        p = min(max(p, 1e-3), 1 - 1e-3)
        return math.log(p / (1 - p))
    w, d, l = (max(v, 1e-12) for v in wdl)
    t, ln = math.log(w) - math.log(l), math.log(d) - (math.log(w) + math.log(l)) / 2
    eL = lg(es)
    ml, md, mw = (min(max(v, 1e-4), 1.0) for v in mv)
    z, f = (E - 2000) / 1000, men / 32
    o, e = qp["odd"], qp["even"]
    t += (o["eL"] * eL + o["eL|eL|"] * eL * abs(eL) + o["T_Maia"] * (math.log(mw) - math.log(ml)) + o["eL*top"] * eL * top + o["eL*ent"] * eL * ent
          + o["eL*z"] * eL * z + o["eL*phase"] * eL * f + o["tempo"])
    ln += (e["1"] + e["|eL|"] * abs(eL) + e["logit D_Maia"] * lg(md) + e["top"] * top + e["ent"] * ent + e["z"] * z + e["phase"] * f
           + e["|eL|*top"] * abs(eL) * top + e["|eL|*ent"] * abs(eL) * ent)
    m = max(t / 2, ln, -t / 2)
    a, b_, c = math.exp(t / 2 - m), math.exp(ln - m), math.exp(-t / 2 - m)
    s_ = a + b_ + c
    return a / s_, b_ / s_, c / s_


def wdl_endgame(path="sf", opp=False):
    """(module, policy kind, parameters of the current time control and venue) of the candidate named by the option Endgame Evaluation,
    path 'sf' (Stockfish values: Fast) or 'leela' (Leela values: Accurate), opp: the opponent's rating given (both ratings set, as Siby
    does); None when the option is Off or the candidate has no parameters there (Blitz online; Accurate outside Classical);
    path None: (module, policy kind, None) whenever the option names a candidate."""
    c = _endgame_cand()
    if c == "Off":
        return None
    global _WE
    if _WE is None:
        try:
            sys.path.insert(0, str(HERE / "nn"))
            import wdl_endgame_p1 as _m
            _m.load_params(str(HERE / "nn" / ENDGAME_PARAMS), default=True)
            _WE = _m
        except Exception as exc:  # noqa: BLE001
            _WE = False
            out(f"info string Toby: Endgame Evaluation unavailable ({type(exc).__name__}: {exc}): Toby 1.0's evaluations without it "
                "(reinstall Toby)")
    if not _WE:
        return None
    cand = _WE.load_params()["candidates"].get(c)
    if not cand:
        return None
    if path is None:
        return (_WE, cand["policy"], None)
    gp = _WE.group_params(c, opts["mode"], venue(), path, opp=opp)
    return (_WE, cand["policy"], gp) if gp else None


def endgame_base(path="sf", opp=False):
    """The refitted base constants of the candidate named by the option Endgame Evaluation for this path and use, or None (Off, or no
    refit for this time control / venue: the engine's own constants)."""
    c = _endgame_cand()
    if c == "Off" or wdl_endgame(None) is None:
        return None
    return _WE.base_consts(c, opts["mode"], venue(), path, opp=opp)


def _x_released(E):
    """The input rating of the released Maia-3 (the study's data): min(E + 700, 2900) in Classical, the rule's x otherwise."""
    return int(round(min(E + 700, 2900))) if opts["mode"] == "Classical" else rule_params(E)[0]


def endgame_policy(board, E, ev, kind, root, ev_other=None, E_other=None):
    """(top, entropy) of Maia's policy as the candidate reads it, for the side to move of board (strength E, evaluation rating ev; the
    other side's ev_other / E_other when the opponent's rating is given): 'valpass' from the value pass (cached; a pass not yet made also
    fills the value cache, so no pass is made twice); 'input' from an extra pass of the released Maia-3 at the input ratings (root rounded
    as the study's data)."""
    if kind == "valpass":
        k = (board.fen(), ev, ev_other)
        if k not in _MVP:
            with MAIA_LOCK:
                _MV[k] = _maia_value_raw(board, ev, ev_other, pol=True)
        return _MVP[k]
    with MAIA_LOCK:
        pm = _maia_policy_raw(board, _x_released(E), _x_released(E_other) if E_other is not None else None, None, maia, None)
    if root:
        pm = {u: round(p, 5) for u, p in pm.items() if p >= 1e-5}
    z = sum(pm.values()) or 1.0
    return max(pm.values()) / z, -sum((v / z) * math.log(v / z) for v in pm.values() if v > 0)


def evals_batch(b, moves, we, qp, ev_elo, ev_opp, opp):
    """The arguments of toby_evals' single Maia value pass: (every position after a listed move that does not end the game, + the
    position itself when Endgame Evaluation applies to it; the ratings; allpol)."""
    after = []
    for u in moves:
        c = b.copy(stack=False)
        c.push(chess.Move.from_uci(u))
        if not c.is_game_over():
            after.append(c)
    if we and not opp and we[0].applies(b.fen()):   # Endgame Evaluation: the position itself in the same batch (same ratings)
        after.append(b)
    return after, ev_opp if opp else ev_elo, ev_elo if opp else None, qp is not None


def evals_setup(E, x, opp, consts, src):
    """toby_evals' ratings and options for its Maia passes: (ev_elo, ev_opp, we, qp), exactly as toby_evals computes them."""
    ev_elo = eval_elo(E, x)
    ev_opp = eval_elo(opp[0], opp[1]) if opp else None
    we = wdl_endgame("sf" if (consts or src == "sf") else "leela", opp is not None)
    cb = endgame_base("sf" if (consts or src == "sf") else "leela", opp is not None)
    if cb is not None and not opp:
        consts = cb
    qp = mix_q("sf" if (consts or src == "sf") else "leela", opp is not None)
    return ev_elo, ev_opp, we, qp


def toby_evals(b, E, x, moves, lee, extra, root, deadline, opp=None, consts=None, src="leela", dist=None):
    """Toby's evaluation of each move, from the point of view of the player who plays it: {uci: (WL, D)} with WL = win - loss, plus that of the position itself.
    The position after the move is evaluated for the side that must play there (the opponent), where the model is calibrated, then flipped.
    opp = (E_opp, x_opp): the opponent's strength and Maia input rating when the user gave it (None: same rating on both sides).
    src: 'sf' when the values come from Stockfish (Fast), 'leela' from Leela's search (Accurate): the Endgame Evaluation parameters.
    dist: Toby's move probabilities (candidate M: the position's W/D/L is their expectation over the evaluated moves)."""
    ev_elo = eval_elo(E, x)
    ev_opp = eval_elo(opp[0], opp[1]) if opp else None
    root_wl = root[0] if root else 0.0
    we = wdl_endgame("sf" if (consts or src == "sf") else "leela", opp is not None)   # option Endgame Evaluation (Off: None)
    cb = endgame_base("sf" if (consts or src == "sf") else "leela", opp is not None)   # a candidate's refitted base constants (else None)
    if cb is not None and not opp:
        consts = cb
    E_o = opp[0] if opp else E   # strength of the side to move after a move
    qp = mix_q("sf" if (consts or src == "sf") else "leela", opp is not None)   # candidate M: the terms Q (else None)
    try:   # every position after a move in one batch (same values as one by one, much faster on a CPU)
        maia_values_batch(*evals_batch(b, moves, we, qp, ev_elo, ev_opp, opp))
    except Exception:  # noqa: BLE001
        pass

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
            mv_c = maia_value_cached(c, ev_opp, ev_elo)
            wo, do, lo = wdl_model(es_opp, mv_c, opp[0], opp[0] - E, consts=cb)
        else:
            mv_c = maia_value_cached(c, ev_elo)
            wo, do, lo = wdl_model(es_opp, mv_c, E, consts=consts)
        if we and we[0].applies(c.fen()):   # Endgame Evaluation: the position after the move, for the opponent (strength E_o)
            tp, en = endgame_policy(c, E_o, ev_opp if opp else ev_elo, we[1], False, ev_elo if opp else None, E if opp else None)
            wo, do, lo = we[0].correct((wo, do, lo), 2 * es_opp - 1, mv_c, tp, en, c.fen(), E_o, we[2], "after")
        if qp is not None:   # candidate M: Q on the move's W/D/L (Classical, Fast, both ratings)
            tq, eq = endgame_policy(c, E_o, ev_opp, "valpass", False, ev_elo, E)
            wo, do, lo = apply_mix_q((wo, do, lo), es_opp, mv_c, tq, eq, E_o, len(c.piece_map()), qp)
        res[u] = (lo - wo, do)  # from the point of view of the player who played the move
    if mix_on() and dist and res:   # candidate M: the position = the expectation of its evaluated moves under Toby's probabilities
        tot = sum(dist.get(u, 0.0) for u in res)
        if tot > 0:
            wl_ = sum(dist.get(u, 0.0) * v[0] for u, v in res.items()) / tot
            d_ = sum(dist.get(u, 0.0) * v[1] for u, v in res.items()) / tot
            return res, (wl_, d_)
    tp = en = None
    if we and we[0].applies(b.fen()):   # Endgame Evaluation: the policy first (its pass also gives the value: no second pass)
        tp, en = endgame_policy(b, E, ev_elo, we[1], True, ev_opp, opp[0] if opp else None)
    if opp:
        mv_b = maia_value_cached(b, ev_elo, ev_opp)
        w, dr, l = wdl_model((1 + root_wl) / 2, mv_b, E, E - opp[0], consts=cb)
    else:
        mv_b = maia_value_cached(b, ev_elo)
        w, dr, l = wdl_model((1 + root_wl) / 2, mv_b, E, consts=consts)
    if tp is not None:   # Endgame Evaluation: the position itself
        w, dr, l = we[0].correct((w, dr, l), root_wl, mv_b, tp, en, b.fen(), E, we[2], "root")
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
    cap = getattr(_MUTE, "capture", None)
    if cap is not None:   # option Deepening: a nested evaluation only records its blocks (the final one gives the position's line)
        cap.append((provisional, root_eval))
        return
    fmt = opts["display"]
    if root_eval:
        ROOT_LAST[:] = [root_eval]
    if evals:
        EV_LAST.update(evals)   # a move keeps its last evaluation when a stage does not re-evaluate it: every listed move then has its WDL line in the same block (Nibbler sorts the moves that were in the latest block first)
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    keep = [(u, p) for u, p in items if p >= 0.001 or u in T2_SHOWN or (fmt == "Nibbler" and u in LISTED)] or items[:1]
    if fmt == "Nibbler":   # a move listed at an earlier stage stays listed (with its new, smaller number): Nibbler keeps the old row, greyed, under the latest block otherwise
        LISTED.update(u for u, _ in keep)
    items = keep if fmt == "Nibbler" else keep[:int(opts["MultiPV"])]  # Nibbler sends its own MultiPV (3): ignored, every move above 0.1% is shown, like Leela
    if EV_LAST:   # only the moves the search has evaluated are listed (each with its W/D/L); before any evaluation, at most as many as it will evaluate
        items = [(u, p) for u, p in items if u in EV_LAST]
    elif EVAL_K[0]:
        items = items[:EVAL_K[0]]
    items = items or keep[:1]
    bd_o = bd if bd is not None else board
    listed_o = {u for u, _ in items}
    others_line = (f"info string toby others {max(0.0, sum(p for u, p in dist.items() if u not in listed_o)):.4f} "
                   f"{max(0, bd_o.legal_moves.count() - len(listed_o))}")   # the legal moves not listed: their total probability and number (Siby)
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
    elif fmt == "Full":   # for interfaces built for Toby (docs/GUI_INTEGRATION.md): one machine-readable line per move, plus standard MultiPV lines
        bd_ = bd if bd is not None else board
        out(f"info string toby begin stage {stage} final {0 if provisional else 1} moves {len(items)} model {'network' if (network_on() or hist_on() or opts['calculator'] == 'Stockfish') else 'rule'}")
        for k, (u, p) in enumerate(items):
            val = EV_LAST.get(u)
            if val:
                wl, d = val
                w, l = (1 - d + wl) / 2, (1 - d - wl) / 2
                ev = f"wdl {int(round(1000 * w))} {int(round(1000 * d))} {int(round(1000 * l))} es {w + d / 2:.4f}"
            else:
                ev = "wdl - - - es -"
            out(f"info string toby move {u} san {bd_.san(chess.Move.from_uci(u))} prob {p:.5f} maia {PRIOR.get(u, 0.0):.5f} {ev}")
            cp = q_to_cp(val[0]) if val else 0
            wdl = f" wdl {int(round(1000 * w))} {int(round(1000 * d))} {int(round(1000 * l))}" if val else ""
            out(f"info depth {depth} seldepth {depth} multipv {k + 1} score cp {cp}{wdl} nodes {nodes} nps {int(nodes / max(ms / 1000, 0.001))} time {ms} pv {u}")
        if root_eval:
            wl, d = root_eval
            w, l = (1 - d + wl) / 2, (1 - d - wl) / 2
            out(f"info string toby position wdl {int(round(1000 * w))} {int(round(1000 * d))} {int(round(1000 * l))} es {w + d / 2:.4f}")
        out(others_line)   # before the block's end line (Siby's "other moves" row is drawn with the block)
        out(f"info string toby end stage {stage} final {0 if provisional else 1}")
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
    if fmt != "Full":
        out(others_line)
    txt = " ".join(f"{bd.san(chess.Move.from_uci(u))} {100 * p:.1f}%" for u, p in items[:6])
    out(f"info string Toby{note}: {txt}")
    if not provisional:   # the search's answer is complete: no heartbeat while a 'go infinite' waits for 'stop'
        HEART["t0"] = None


# ----------------------------------------------------------------------------------------------------- search
DRAW_ES = float(os.environ["TOBY_DRAW_ES"]) if os.environ.get("TOBY_DRAW_ES") else None
# Off by default (2026-10-27): in real games, players who can end the game by repetition do it far more often than a strict "never above 50%"
# rule would (nn/repetition/: 35,000 positions; even at 70% expected score or more, 30 to 60% of players repeat). Set TOBY_DRAW_ES=0.5 for a
# more combative bot: a move that ends the game in a draw is then worth exactly that expected score to the side to move.


def ends_in_draw(b, u):
    """True if playing u ends the game in a draw: stalemate, threefold repetition (from the move history the GUI sent), 50-move rule,
    insufficient material."""
    b.push(chess.Move.from_uci(u))
    try:
        return b.is_stalemate() or b.is_repetition(3) or b.halfmove_clock >= 100 or b.is_insufficient_material()
    finally:
        b.pop()


def pick_move(dist, b=None):
    """Most likely move by default; random draw from the probabilities only in the (hidden) human mode, meant for a future bot build.
    Draws (only when TOBY_DRAW_ES is set; off by default): a move that ends the game in a draw is worth exactly DRAW_ES to the side to move. When Toby's expected score for the side to
    move is above that, it does not play such a move: it plays the most likely move among those it rates above DRAW_ES (if there is one).
    Below DRAW_ES, the draw is welcome and the move stays. Only the move played changes, never the probabilities shown."""
    human = opts["move_choice"] == "human"

    def choose(d):
        if human:
            us = list(d)
            lam = play_lambda(PLAY_LOSS["E"]) if PLAY_LOSS.get("loss") else 0.0
            ws = [d[u] * (math.exp(-lam * PLAY_LOSS["loss"].get(u, 0.0)) if lam else 1.0) for u in us]
            return random.choices(us, weights=ws)[0]
        return max(d, key=d.get)
    u = choose(dist)
    if DRAW_ES is None or b is None or not ROOT_LAST or not ends_in_draw(b, u):
        return u
    wl, _ = ROOT_LAST[0]
    if (1 + wl) / 2 <= DRAW_ES:
        return u
    better = {v: p for v, p in dist.items() if v in EV_LAST and (1 + EV_LAST[v][0]) / 2 > DRAW_ES and not ends_in_draw(b, v)}
    if not better:
        return u
    v = choose(better)
    out(f"info string Toby: {b.san(chess.Move.from_uci(u))} would end the game in a draw (50%) while Toby rates the position at "
        f"{100 * (1 + wl) / 2:.0f}%: plays {b.san(chess.Move.from_uci(v))} instead")
    return v


FIDE_LOWEST = 1400      # lowest rating FIDE publishes
CONVERSION_FLOOR = 1660  # lowest FIDE value of the conversion tables (ChessGoals): below, the FIDE equivalent of a platform rating is a convention


def own_rating(b):
    """Rating entered for the side to move (as side_ratings reads the options)."""
    wr, br = opts["white_rating"], opts["black_rating"]
    if wr > 0 and br > 0:
        return wr if b.turn == chess.WHITE else br
    return wr or br or DEFAULT_RATING


def strength_text(E):
    """E as shown to the user. A FIDE rating is shown as it is; the FIDE equivalent of a platform rating below the conversion data is not
    data but a convention (toby_rating_convert.to_fide): shown as 'E <= 1400' below FIDE's lowest rating, and marked as a convention between."""
    if opts["source"] == "FIDE" or E >= CONVERSION_FLOOR:
        return f"E = {E:.0f}"
    if E <= FIDE_LOWEST:
        return f"E <= {FIDE_LOWEST} (FIDE-equivalent; no conversion data below {CONVERSION_FLOOR})"
    return f"E = {E:.0f} (by convention: no conversion data below {CONVERSION_FLOOR})"


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


MAIA_ONLY_BELOW = float(os.environ.get("TOBY_MAIA_ONLY_BELOW", "0.4"))   # seconds: with less time than this for the move, Maia alone answers at once
ZEITNOT_CLOCK = float(os.environ.get("TOBY_ZEITNOT", "10"))              # seconds left on the clock: below, Maia alone (deep time trouble)


def time_budget(t, turn):
    """Seconds Toby may use for this move, from the go command (None = no limit: analysis, go infinite, or no clock given).
    movetime N: N minus a small margin. Clocks: about remaining / moves to go (30 by default) + 3/4 of the increment, at most a quarter of the clock;
    a clock below ZEITNOT_CLOCK seconds gives 0 (Maia alone). The search uses the budget by stages: Maia (instant), Leela's static values (about 0.5 s),
    the full rule (2-5 s); when the budget runs out, the last complete stage is played."""
    def num(k):
        return float(t[t.index(k) + 1]) if k in t and t.index(k) + 1 < len(t) else None
    if "infinite" in t:
        return None
    if num("movetime") is not None:
        return max(0.0, num("movetime") / 1000 - 0.15)
    mine, inc = ("wtime", "winc") if turn == chess.WHITE else ("btime", "binc")
    left = num(mine)
    if left is None:
        return None
    left /= 1000
    if left < ZEITNOT_CLOCK:
        return 0.0
    mtg = num("movestogo") or 30
    inc_s = (num(inc) or 0) / 1000
    return max(0.0, min(left / mtg + 0.75 * inc_s, 0.25 * left) - 0.15)


def go_clock(t, turn):
    """(mover's seconds, opponent's seconds) from 'go wtime W btime B', or None (analysis, movetime only)."""
    try:
        w, bl = float(t[t.index("wtime") + 1]) / 1000, float(t[t.index("btime") + 1]) / 1000
    except (ValueError, IndexError):
        return None
    return (w, bl) if turn == chess.WHITE else (bl, w)


BEST_SENT = [0]   # number of bestmove lines sent (search_guarded)


def search_guarded(movetime, infinite=False):
    """The search thread: any exception is said in an info string and the search still ends with a bestmove (the most probable move
    known, else the first legal move): a GUI never waits forever."""
    n0 = BEST_SENT[0]
    try:
        search(movetime, infinite)
    except BaseException as exc:  # noqa: BLE001
        try:
            import traceback
            where = traceback.extract_tb(exc.__traceback__)[-1]
            where = f" at {Path(where.filename).name}:{where.lineno}"
        except Exception:  # noqa: BLE001
            where = ""
        try:
            out(f"info string Toby ERROR: the search failed ({type(exc).__name__}: {exc}{where})")
        except Exception:  # noqa: BLE001
            pass
        if BEST_SENT[0] == n0:
            try:
                b = board.copy()
                legal = [m.uci() for m in b.legal_moves]
                cand = [u for u, _ in sorted(PRIOR.items(), key=lambda kv: -kv[1]) if u in legal]
                out(f"bestmove {(cand or legal or ['0000'])[0]}")
            except Exception:  # noqa: BLE001
                out("bestmove 0000")


def search(movetime, infinite=False):
    """movetime = time budget in seconds (time_budget) or None: no limit (analysis)."""
    timer = None
    if movetime is not None and movetime >= MAIA_ONLY_BELOW:   # the budget runs out: the search stops and the last complete stage is played
        timer = threading.Timer(movetime, stop_flag.set)
        timer.daemon = True
        timer.start()
    try:
        _search(movetime, infinite)
    finally:
        if timer is not None:
            timer.cancel()


def _deepen_w():
    """Option Deepening: the weight of the next position's own evaluation in a listed move's W/D/L (nn/deepening_params.json; fitted on
    the game results, the method page)."""
    try:
        return float(json.loads((HERE / "nn" / "deepening_params.json").read_text(encoding="utf-8"))["w"])
    except Exception:  # noqa: BLE001
        return 0.55


DEEPEN_W = _deepen_w()


def evaluate_position(b):
    """Toby's displayed line of position b (W - L, D for its side to move), the engine's own final answer with b's history and the current
    options, computed silently (option Deepening). None if stopped or no line."""
    saved = (dict(PRIOR), dict(EV_LAST), list(ROOT_LAST), set(LISTED), EVAL_K[0], dict(PLAY_LOSS), set(T2_SHOWN))
    prec = (opts.get("hybrid"), opts["calculator"])
    opts["hybrid"], opts["calculator"] = True, "Stockfish"   # the next positions are evaluated in Fast (hybrid), also under Accurate (Yosha, 10 Nov)
    _MUTE.on, _MUTE.capture = True, []
    try:
        _search(None, False, bd=b, nested=True)
        fin = [r for prov, r in _MUTE.capture if not prov and r]
        return fin[-1] if fin else None
    finally:
        cap = _MUTE.capture
        _MUTE.on, _MUTE.capture = False, None
        if (opts.get("hybrid"), opts["calculator"]) == (True, "Stockfish"):   # restored unless a setoption changed them meanwhile
            opts["hybrid"], opts["calculator"] = prec
        PRIOR.clear(); PRIOR.update(saved[0]); EV_LAST.clear(); EV_LAST.update(saved[1]); ROOT_LAST[:] = saved[2]
        LISTED.clear(); LISTED.update(saved[3]); EVAL_K[0] = saved[4]; PLAY_LOSS.clear(); PLAY_LOSS.update(saved[5])
        T2_SHOWN.clear(); T2_SHOWN.update(saved[6])


def deepen(b, dist, t0):
    """Option Deepening (go infinite, after the final block): each of the 3 most probable listed moves, plus the listed move with the best
    move-view ES when it is not among them (amendment 15), gets W/D/L = (1 - w) its move view + w the next position's own displayed line
    (flipped), w = DEEPEN_W; the position = sum p x listed W/D/L; ONE updated final block."""
    listed = [u for u, _ in sorted(dist.items(), key=lambda kv: -kv[1]) if u in EV_LAST]
    if not listed:
        return
    to_wdl = lambda wl, d: ((1 - d + wl) / 2, d, (1 - d - wl) / 2)
    spec = y60_spec()
    raw = dict(Y60_RAW) if spec is not None else {}   # this search's children, before the nested searches overwrite Y60_RAW
    dbg_in, dbg_nest = {}, {}   # TOBY2_DUMP only: each blend's move-view input and the nested line
    upd = {}
    best = max(listed, key=lambda u: EV_LAST[u][0])   # the best move view (ES = (1 + W - L) / 2); first in probability order on a tie
    nexts = []
    for u in listed[:3] + ([best] if best not in listed[:3] else []):
        c = b.copy()
        c.push(chess.Move.from_uci(u))
        nexts.append((u, c, c.is_game_over()))
    for k, (u, c, over) in enumerate(nexts):
        if stop_flag.is_set():
            return
        if over:
            continue
        if t2_on():   # pipelining: the next position's body pass during this one's Stockfish complements (output-identical)
            later = [c2 for _, c2, o2 in nexts[k + 1:] if not o2]
            if later:
                try:
                    t2_queue_body(later[0])
                except Exception:  # noqa: BLE001
                    pass
        r = evaluate_position(c)
        if stop_flag.is_set():
            return
        if r is None:   # no line there (search stopped or failed): the move keeps its move view
            continue
        wm, dm, lm = to_wdl(*EV_LAST[u])
        if u in raw and raw[u][1] is not None:   # Y60-half: the deepened move's own view at v x deepened_scale (the shown one chose it)
            wm, dm, lm = y60_correct(raw[u][0], raw[u][1], raw[u][2], raw[u][3], spec, scale=float(spec["deepened_scale"]))
        dbg_in[u], dbg_nest[u] = [wm, dm, lm], list(r)
        lc, dc, wc = to_wdl(*r)   # the next position's line is for its side to move: flipped
        w_, d_, l_ = ((1 - DEEPEN_W) * wm + DEEPEN_W * wc, (1 - DEEPEN_W) * dm + DEEPEN_W * dc, (1 - DEEPEN_W) * lm + DEEPEN_W * lc)
        upd[u] = (w_ - l_, d_)
    if stop_flag.is_set():
        return
    vals = {u: upd.get(u, EV_LAST[u]) for u in listed}
    tot = sum(dist[u] for u in listed)
    W = sum(dist[u] * to_wdl(*vals[u])[0] for u in listed) / tot
    D = sum(dist[u] * to_wdl(*vals[u])[1] for u in listed) / tot
    L_ = sum(dist[u] * to_wdl(*vals[u])[2] for u in listed) / tot
    if os.environ.get("TOBY2_DUMP"):   # development / checks: the deepened block's exact numbers (offline recomputation of Deepening)
        try:
            with open(os.environ["TOBY2_DUMP"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"deepened": {"fen": b.fen(), "listed": listed, "p": {u: dist[u] for u in listed},
                                                  "views": {u: list(EV_LAST[u]) for u in listed}, "chosen": [u for u, _, _ in nexts],
                                                  "inputs": dbg_in, "nested": dbg_nest, "upd": {u: list(v) for u, v in upd.items()},
                                                  "position": [W, D, L_], "w": DEEPEN_W}}) + "\n")
        except Exception as exc:  # noqa: BLE001
            out(f"info string toby deepening dump failed ({exc})")
    emit(3, dist, {}, {}, None, 1, t0, False, evals=upd, root_eval=(W - L_, D), bd=b)
    out("info string toby deepened")



# ----------------------------------------------------------------------------------------------------- Toby 2 move model (network B)
# toby2.py: network B (Leela BT4 body fine-tuned on human games, move head + W/D/L head) with Stockfish 19 values by Toby 1.0's recipe
# (V2_BASE nodes on every move, V2_EXT on the candidates). Option Move Network = Toby 2 (default when the weights file is found) replaces
# Maia + the corrector for the move distribution; Toby 1.0's surrounding modules stay, each switchable: Rating Source conversion, the
# per-move W/D/L (Move Evaluations: Toby 2 = network B on each position after a move, the default since 2.0.2 / Toby 1.0 = Toby 1.0's
# module, with Endgame Evaluation / Off), the position's W/D/L (Position Evaluation: Toby 2 = sum over the moves of Toby 2's probability x
# Toby 2's W/D/L after the move, the default since 2.0.2 / Network = B's W/D/L head / Toby 1.0 / Mix), Deepening, the Human-mode blunder
# check. No silent fallback: a missing or failed network is said, and Toby 1.0 answers.
MOVE_NETWORKS = ["Toby 2", "Toby 1.0"]
T2_DEVICES = ["Auto", "CPU", "GPU"]
MOVE_EVALS = ["Toby 2", "Toby 1.0", "Off"]   # option Move Evaluations ("true" = Toby 2, "false" = Off: the check option of 2.0 / 2.0.1)
POSITION_EVALS = ["Toby 2", "Network", "Toby 1.0", "Mix"]
# "Move Evaluations = Toby 1.0" with "Position Evaluation = Network" gives exactly the 2.0.1 answer (W/D/L included).
T2 = {"net": None, "error": None, "loading": None, "search": None, "eng": None, "cfg": None, "lc0": None, "leela": None, "leela_error": None}
T2_PRODUCTS = ["Standard", "Accurate"]   # one product file each (toby2_standard.json / toby2_accurate.json, next to the network files).
# Standard = the sealed protocol's "Toby 2 Fast" (label only). Accurate is GPU-only: offered only with an NVIDIA GPU and a lc0 next to
# Toby, and used only when the probe passes (otherwise Standard answers, said once and in the path line). Toby 1.0 keeps its own names.
T2_HEADS = ["Product", "On", "Off"]  # Leela head (Accurate only): Product = the product file's rule per cadence (classical on; rapid / blitz
                                     # by the sealed protocol's rule 6.3), On / Off = forced (scoring H-rb / A-FB)
T2_GO_INC = [None]   # the mover's increment (s) of the last 'go winc / binc', used with the GUI's clocks
T2_VALUES_BELOW = float(os.environ.get("TOBY2_VALUES_BELOW", "0.6"))   # seconds for the move: below, B alone with its value inputs masked
# Base time / increment when the user sets none (options Base Time = 0, Increment = -1): unknown, the masked path. Measured on 3,000 classical
# dev positions with the stand-in network (nn/v7/toby2_check.py clocks): clocks and base / increment masked +0.0072 [+0.0042; +0.0107] LL
# against the true values; clocks masked with the true base +0.0015 (n.s.); clocks masked with 90 min + 30 s +0.0009 (n.s.). To re-check
# with the final network. With 'go wtime btime' the GUI's clocks are used (increment from winc / binc).
# The third Stockfish (output-identical speculative complements, toby2.Toby2Search spec3). TOBY2_SPEC3_MIN_CORES (development, the old
# rule): from this many CPUs in the affinity mask, 0 = never. Otherwise automatic: on when usable_cores() minus t2_cores_used() leaves at
# least T2_SPEC3_FREE cores. On the A40 pod (7.65-core quota, 3.65 or 2.65 free) it slowed the body more than it saved (0.594 vs 0.565-0.570 s):
# off there.
T2_SPEC3_MIN_CORES = int(os.environ["TOBY2_SPEC3_MIN_CORES"]) if os.environ.get("TOBY2_SPEC3_MIN_CORES") else None
T2_SPEC3_FREE = float(os.environ.get("TOBY2_SPEC3_FREE", "16"))   # measured (the method page): no gain with 2.65-3.65 free (A40), 4 or 12 free (CPU pod)
T2_SF_THREADS = int(os.environ.get("TOBY2_SF_THREADS", "1"))   # development speed study only: != 1 changes the values (lesson P2), said in the path line
T2_MIX_W = 0.5   # Position Evaluation = Mix: W/D/L = (1 - w) network + w Toby 1.0 (test T1, nn/v7/TOBY1_TO_TOBY2.md section 4)
try:
    import toby2 as _T2MOD
    if WThread is PooledThread:   # toby2.py's per-search threads (body, speculation, Leela) reuse workers too (its file is unchanged)
        _T2MOD.threading = _ThreadingForToby2()
    T2_WEIGHTS = _T2MOD.default_weights()
except Exception as _exc:  # noqa: BLE001
    _T2MOD, T2_WEIGHTS = None, None
    T2["error"] = f"toby2 module: {_exc}"
opts.update({"t2_product": {"Fast": "Standard"}.get(os.environ.get("TOBY2_PRODUCT", "Standard"), os.environ.get("TOBY2_PRODUCT", "Standard")),
             "t2_head": "Product", "t2_candidates_set": False})
opts.update({"move_network": "Toby 2" if (T2_WEIGHTS is not None and T2_WEIGHTS.exists()) else "Toby 1.0",
             "t2_candidates": os.environ.get("TOBY2_CANDIDATES", "Maia C"), "t2_device": os.environ.get("TOBY2_DEVICE", "Auto"),
             "t2_threads": int(os.environ.get("TOBY2_THREADS", "0")), "t2_move_evals": os.environ.get("TOBY2_MOVE_EVALS", "Toby 2"),
             "t2_position_eval": os.environ.get("TOBY2_POSITION_EVAL", "Toby 2"),
             "t2_blunder_check": True, "t2_base": 0, "t2_inc": -1})


def t2_config_path():
    """The product file: TOBY2_CONFIG (one file, the product is then fixed), else toby2_<product>.json next to the network file. None if absent
    (development copies without product files: the network file alone, said in the path line)."""
    env = os.environ.get("TOBY2_CONFIG")
    if env:
        return Path(env)
    f = t2_config_dir() / f"toby2_{opts['t2_product'].lower()}.json"
    return f if f.exists() else None


def t2_config_dir():
    return Path(os.environ.get("TOBY2_CONFIG_DIR") or (T2_WEIGHTS.parent if T2_WEIGHTS is not None else HERE))


def t2_accurate_offered():
    """The option Toby 2 Product lists Accurate only where it can run: its product file, an NVIDIA GPU answering nvidia-smi and a lc0
    file (the light package has no lc0). The full probe (sm_86, hashes, one lc0 start) runs when Accurate is chosen."""
    cfgf = Path(os.environ["TOBY2_CONFIG"]) if os.environ.get("TOBY2_CONFIG") else t2_config_dir() / "toby2_accurate.json"
    try:
        acc = cfgf.exists() and bool(json.loads(cfgf.read_text(encoding="utf-8")).get("head"))
    except Exception:  # noqa: BLE001
        acc = False
    return acc and HAS_NVIDIA and bool(os.path.exists(LC0) or shutil.which(LC0))


def t2_device():
    d = opts["t2_device"]
    cfg = T2.get("cfg")
    if d == "Auto" and cfg is not None:   # the product file's device (Fast: the CPU; Accurate: the GPU)
        d = {"cpu": "CPU", "cuda": "GPU"}.get(cfg.get("device"), "Auto")
    if d == "CPU":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    if d == "GPU":
        out("info string Toby 2: no NVIDIA GPU answered, network on the processor")
    return "cpu"


T2_LOAD_LOCK = threading.RLock()   # one load of network B at a time (the background load at launch, isready, a search)


def load_t2():
    """Loads network B once (about 3-10 s) with the product file's settings; errors are kept and said at each search, never hidden."""
    with T2_LOAD_LOCK:
        if T2["net"] is not None and T2.get("loaded_for") not in (None, opts["t2_product"]) and not os.environ.get("TOBY2_CONFIG"):
            T2.update(net=None, search=None, cfg=None, error=None)   # the product changed while the background load ran: the asked one
        return _load_t2()


def t2_prepare():
    """isready: everything the first Toby 2 search would otherwise do first, so that it answers at once: network B loaded with the
    current options, for Accurate the GPU / lc0 probe settled (or its wait run out: the next search does not wait again), the network
    reloaded if the device changed, the Stockfish engines started. The same calls as at the start of _search_t2."""
    if not t2_on() or os.environ.get("TOBY2_ISREADY_LOAD", "1") == "0":   # 0: the old behaviour (network B loaded at the first search)
        return
    net = load_t2()
    if net is not None and T2.get("cfg") is not None and T2["cfg"].get("head"):
        if t2_probe_settle() and T2.get("standard_after_probe"):
            T2["standard_after_probe"] = False
            net = load_t2()
    if net is not None and net.device != t2_device():
        T2.update(net=None, search=None)
        net = load_t2()
    if net is not None:   # Stockfish started from a daemon thread, as when a search starts it: python-chess's engine threads then do
        th_ = threading.Thread(target=t2_searcher, daemon=True)   # not keep the process alive after 'quit'
        th_.start()
        th_.join()


def t2_say_missing():
    """Once: Toby 2 is the default but its network file (or module) is missing, so Toby 1.0 answers (said, never silent)."""
    if T2.get("said_missing") or t2_on() or (opts.get("move_network_set") and opts["move_network"] != "Toby 2"):
        return   # (a user who chose Toby 1.0 is not told)
    T2["said_missing"] = True
    if _T2MOD is None:
        out(f"info string Toby 2 unavailable ({T2['error']}): {T1_ANSWERS}")
    else:
        out(f"info string Toby 2 network not found ({T2_WEIGHTS}): {T1_ANSWERS}")


def _load_t2():
    if T2["net"] is not None or T2["error"] or _T2MOD is None:
        return T2["net"]
    prod0 = opts["t2_product"]
    try:
        t = time.time()
        cp = t2_config_path()
        if cp is None and opts["t2_product"] == "Accurate" and not os.environ.get("TOBY2_CONFIG"):   # e.g. the light package: no
            sp_ = t2_config_dir() / "toby2_standard.json"                                         # Accurate product file, no lc0
            if sp_.exists():
                why = "no lc0 in this package" if not (os.path.exists(LC0) or shutil.which(LC0)) else "no toby2_accurate.json in this package"
                T2["accurate_unavailable"] = T2["accurate_unavailable"] if T2.get("accurate_unavailable") else why
                if not T2.get("said_unavailable"):
                    T2["said_unavailable"] = True
                    out(f"info string Toby 2 Accurate is not available here ({why}): Toby 2 Standard answers")
                cp = sp_
        if cp is not None and T2.get("accurate_unavailable"):   # Accurate already found unusable in this process: Standard, for good
            cp = Path(cp).parent / "toby2_standard.json"
        cfg = T2["cfg"] = _T2MOD.load_config(cp) if cp is not None else None   # raises on a hash mismatch: Toby 2 is then unavailable (said)
        y60_check(cfg)   # a malformed wdl_correction field: Toby 2 unavailable, said (never a silent guess)
        if cfg is not None and cfg.get("head") and cfg.get("leela") and T2["leela"] is None and not T2["leela_error"]:   # Accurate is GPU-only: the probe (GPU <= 5 s, lc0 start <= 10 s,
            # self-check ~1-3 s) runs in the background while network B loads; a quick failure (no GPU visible, nvidia-smi) is known at once
            th_ = T2["probe"] = threading.Thread(target=t2_leela, daemon=True)
            th_.start()
            th_.join(timeout=1.0)
            if not th_.is_alive() and T2["leela"] is None:
                cfg = t2_to_standard(cp)
        weights, head = T2_WEIGHTS, None
        if cfg is not None:
            weights = Path(cfg["network"]["path"])
            head = cfg["head"]["path"] if cfg.get("head") else None
            if not opts["t2_candidates_set"]:
                opts["t2_candidates"] = cfg["candidates"]
        dev = t2_device()
        dtype = (cfg or {}).get("compute", {}).get(dev, {}).get("dtype") or ("fp16" if dev == "cuda" and os.environ.get("TOBY2_FP16") == "1" else "fp32")
        th = opts["t2_threads"] or (2 if n_cores() >= 4 else 1)   # 4 threads in all: 2 for B, one per Stockfish (base, extension)
        T2["net"] = _T2MOD.Toby2Net(weights, device=dev, dtype=dtype, threads=th, head=head)
        T2["loaded_for"] = prod0   # also on the GPU: torch's default (one thread per visible core) oversubscribes the CPU quota
        STARTUP["toby2"] = time.time()
        out(f"info string toby timing toby2 network loaded {time.time() - t:.2f}s ({T2['net'].label})")
    except Exception as exc:  # noqa: BLE001
        T2["error"] = f"{type(exc).__name__}: {exc}"
    return T2["net"]


def t2_on():
    return opts["move_network"] == "Toby 2" and _T2MOD is not None and T2_WEIGHTS is not None and T2_WEIGHTS.exists()


def t2_head_wanted():
    """The Leela head for this search: Accurate product only; classical by the product file (on), rapid / blitz by its rule (6.3 of the
    sealed protocol: off = B-rb, on = H-rb); option Leela Head = On / Off forces it."""
    cfg, net = T2.get("cfg"), T2.get("net")
    if cfg is None or not cfg.get("head") or net is None or net.head is None:
        return False
    if opts["t2_head"] != "Product":
        return opts["t2_head"] == "On"
    rule = cfg["head_rule"]["classical" if opts["mode"] == "Classical" else "rapid_blitz"]
    return rule == "on"


T2_LC0_START_S = float(os.environ.get("TOBY2_LC0_START_TIMEOUT", "10"))    # lc0 start + network load, at most (then: said, head off)
T2_LC0_SEARCH_S = float(os.environ.get("TOBY2_LC0_SEARCH_TIMEOUT", "20"))  # one 1,024-node search, at most (~0.3 s on an A40)
T2_LEELA_LOCK = threading.Lock()


def t2_leela():
    """The Accurate product's Leela input path (LeelaInputs on its own lc0 process): probed and started ONCE per process (GPU probe <= 5 s,
    lc0 start <= TOBY2_LC0_START_TIMEOUT s, started in the background when the Accurate network loads); the decision is cached. None with
    the reason kept in T2['leela_error'] when lc0 cannot be used here (no GPU visible, nvidia-smi failing, card not Ampere sm_86, lc0 /
    BT3 missing or not the product file's, lc0 hanging or exiting at start, or a failed search later: lc0 is then stopped and never
    relaunched): said at every search and recorded in the path line, the answer is then B's with Stockfish values (never silent)."""
    with T2_LEELA_LOCK:
        if T2["leela"] is not None or T2["leela_error"]:
            return T2["leela"]
        lz = T2["cfg"]["leela"]
        t0 = time.time()
        try:
            ok, gpu = _T2MOD.gpu_for_lc0()
            if not ok:
                raise RuntimeError(gpu)
            exe = shutil.which(LC0) or LC0
            if not os.path.exists(exe):
                raise _T2Unavailable("no lc0 in this package")
            if not os.path.exists(NET):
                raise RuntimeError(f"lc0 or its network not found ({exe}, {NET})")
            if os.environ.get("TOBY2_LC0_NO_HASH") != "1":   # development tests of the start failures only (fake lc0); said below
                plat = "windows" if os.name == "nt" else ("macos" if sys.platform == "darwin" else "linux")
                want_lc0 = lz.get("lc0_sha256")
                want_lc0 = want_lc0.get(plat) if isinstance(want_lc0, dict) else want_lc0   # per platform; None = not pinned there (self-check only)
                for what, f, want in (("lc0", exe, want_lc0), ("Leela network", NET, lz.get("network_sha256"))):
                    if want and _T2MOD.sha256(f) != want:
                        raise RuntimeError(f"{what} {Path(f).name} is not the product file's ({want[:16]})")
            else:
                out("info string Toby 2 Accurate: lc0 / BT3 hash check skipped (TOBY2_LC0_NO_HASH=1, development only)")
            try:
                T2["lc0"] = Lc0(backend=lz["backend"], start_timeout=T2_LC0_START_S, search_timeout=T2_LC0_SEARCH_S)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"lc0 did not start within {T2_LC0_START_S:.0f} s or exited ({exc})") from None
            li = _T2MOD.LeelaInputs(T2["lc0"], lz["backend"], int(lz["nodes"]))
            if lz.get("selfcheck"):   # the card is accepted on evidence: 3 fixed positions against the stored A40 features
                # The probe runs in the background (from isready / the first Accurate search), when the UCI stop flag may be set: 'ucinewgame'
                # sets it until the next 'go'. Its searches therefore obey their own event, never set: a 'stop' or 'ucinewgame' meant for a search
                # must not cut the self-check short (it would see no visits and refuse a good card). The watchdog still bounds each search.
                T2["lc0"].stop_event = threading.Event()
                try:
                    ok, chk = _T2MOD.lc0_selfcheck(li, lz["selfcheck"])
                finally:
                    T2["lc0"].stop_event = None
                if not ok:
                    raise RuntimeError(f"{gpu}: {chk}")
                gpu = f"{gpu}; {chk}"
            if T2.get("accurate_unavailable"):   # the probe finished after its wait ran out: Standard already answers, lc0 is not kept
                raise RuntimeError(T2["accurate_unavailable"])
            T2["leela"] = li
            T2["leela"].gpu = gpu
            T2["leela_gpu"] = gpu
            out(f"info string toby timing toby2 lc0 started {time.time() - t0:.2f}s ({lz['backend']}, {gpu})")
        except Exception as exc:  # noqa: BLE001
            if not T2["leela_error"]:
                T2["leela_error"] = str(exc) if isinstance(exc, _T2Unavailable) else f"{type(exc).__name__}: {exc}"
            t2_leela_stop()
        return T2["leela"]


class _T2Unavailable(RuntimeError):
    """A reason Accurate cannot run here, said as it is (no exception name)."""


def t2_to_standard(cp):
    """Accurate is not available on this machine: Toby 2 Standard's product file (next to Accurate's), said once, in every path line."""
    T2["accurate_unavailable"] = T2["leela_error"]
    sp = Path(cp).parent / "toby2_standard.json"
    if not T2.get("said_unavailable"):
        T2["said_unavailable"] = True
        out(f"info string Toby 2 Accurate is not available here ({T2['leela_error']}): Toby 2 Standard answers")
    if not sp.exists():
        raise RuntimeError(f"Accurate unavailable and no {sp.name} next to it")
    cfg = T2["cfg"] = _T2MOD.load_config(sp)
    return cfg


T2_PROBE_WAIT_S = float(os.environ.get("TOBY2_PROBE_WAIT", "12"))   # a first Accurate search waits at most this for the probe


def t2_probe_settle():
    """Called by each search of the Accurate product: waits (at most T2_PROBE_WAIT_S) for the background probe; if it failed, Toby 2
    Standard takes over for the session (network reloaded with Standard's product file, said once). Returns False while it still runs
    (that answer: head off, said)."""
    th_ = T2.get("probe")
    if th_ is not None and th_.is_alive():
        if T2.get("probe_waited"):   # the probe already had its wait once: no second wait; Toby 2 Standard answers for the session
            if not T2["leela_error"]:
                T2["leela_error"] = f"the GPU / lc0 probe did not finish within {T2_PROBE_WAIT_S:.0f} s"
        else:
            th_.join(timeout=T2_PROBE_WAIT_S)
            if th_.is_alive():
                T2["probe_waited"] = True
                return False
    if T2["leela"] is None and T2.get("cfg") is not None and T2["cfg"].get("head") and not T2.get("accurate_unavailable"):
        cp = Path(os.environ["TOBY2_CONFIG"]) if os.environ.get("TOBY2_CONFIG") else t2_config_dir() / "toby2_accurate.json"
        T2.update(net=None, search=None)
        T2["error"] = None
        t2_to_standard(cp)
        T2["standard_after_probe"] = True
    return True


def t2_leela_stop():
    """lc0 of Toby 2 stopped for good (a failed start or search): never relaunched in this process."""
    lc = T2.get("lc0")
    T2["lc0"] = T2["leela"] = None
    if lc is not None:
        try:
            lc.quit()
        except Exception:  # noqa: BLE001
            pass


def t2_leela_failed(reason):
    with T2_LEELA_LOCK:
        if not T2["leela_error"]:
            T2["leela_error"] = f"search failed earlier in this session ({reason}); lc0 stopped, not relaunched"
        t2_leela_stop()


def t2_maia_top3(b):
    E, Eo = _T2MOD._E_box
    _FT.force = True
    try:
        pm = maia_policy(b, rule_params(E)[0], rule_params(Eo)[0], history=False)   # as the training values (toby_values2: position alone)
    finally:
        _FT.force = None
    return maia_top3(pm, b)


def t2_spec3_wanted():
    """Whether this process runs the third Stockfish: TOBY2_SPEC3_MIN_CORES if set (old rule), else enough free cores (T2_SPEC3_FREE)."""
    if T2_SPEC3_MIN_CORES is not None:
        return T2_SPEC3_MIN_CORES > 0 and n_cores() >= T2_SPEC3_MIN_CORES
    return T2_SF_THREADS == 1 and usable_cores() - t2_cores_used() >= T2_SPEC3_FREE


def t2_searcher():
    if T2["search"] is None:
        if T2["eng"] is None and os.path.exists(SF_PATH):
            T2["eng"] = StockfishCalc(fixed_threads=T2_SF_THREADS)   # 1 thread, Hash 64 MB cleared before every search: the training values' settings
        if T2.get("spec3") is None and t2_spec3_wanted() and os.path.exists(SF_PATH):
            try:   # a third single-thread Stockfish: speculative complements on B's top 3 + pass 1's provisional top 2 (output-identical)
                T2["spec3"] = StockfishCalc(fixed_threads=1)
            except Exception:  # noqa: BLE001
                T2["spec3"] = None
        T2["search"] = _T2MOD.Toby2Search(T2["net"], T2["eng"], spec_engine(), V2_BASE, V2_EXT, maia_top3=t2_maia_top3, spec_lock=_SPEC["busy"],
                                          cand_pass=(T2.get("cfg") or {}).get("candidate_pass", "masked"), spec3=T2.get("spec3"))
    return T2["search"]


def t2_side(net, child=False):
    """B's side inputs of the current search: base / increment from the options (else the GUI's increment with its clocks), the clocks of
    'go wtime btime' (analysis: unknown, the masked-clock path, as the training rows seen without clocks), cadence and venue.
    child=True: those of the positions after a move (Toby 2's per-move W/D/L): the same, with the two clocks as at the root seen from
    the side to move there (the root's opponent)."""
    if CLOCK is not None:
        clk, clko = CLOCK
    else:
        clk = clko = None
    if child:
        clk, clko = clko, clk
    base = float(opts["t2_base"]) if opts["t2_base"] else None
    inc = float(opts["t2_inc"]) if opts["t2_inc"] >= 0 else (T2_GO_INC[0] if CLOCK is not None else None)
    return net.side_inputs(base, inc, clk, clko, _T2MOD.TIME_CONTROL_CADENCE[opts["mode"]], venue())


def t2_queue_body(c):
    """Deepening: the body pass of the next position c (searched right after the current one) is made during the current search's wait
    for its complements (toby2.Toby2Search.prefetch_body, output-identical: used only if that search's inputs are exactly these). Only on
    the GPU, or on the processor with free cores for it."""
    S, net = T2.get("search"), T2.get("net")
    if not T2_BODY_PREFETCH or S is None or net is None or S.net is not net:
        return
    if not (str(net.device).startswith("cuda") or usable_cores() - t2_cores_used() - (1 if T2.get("spec3") is not None else 0) >= T2_PREFETCH_CPU_FREE):
        return
    E, opp = side_ratings(c)
    src = opts["t2_candidates"]
    if src == "Maia C" and maia_ft is None:
        src = "Network"
    S.next_body = (c, E, opp[0] if opp else E, t2_side(net), src)


T2_BODY_PREFETCH = os.environ.get("TOBY2_BODY_PREFETCH", "1") != "0"   # Deepening's body pipelining (output-identical; 0 = off, for checks)


def _sf_pseudo_leela(sf):
    """nn/corrector.sf_pseudo_leela(sf) (Stockfish values as Leela-like (N, WL, D, Q) per move, root = the best move's), from the module
    when it is there, else this copy of the same code (a package without nn/corrector.py must still answer)."""
    if _NNC is not None:
        return _NNC.sf_pseudo_leela(sf)
    lee = {}
    for u, cp in sf.items():
        q = max(-0.999, min(0.999, math.atan(cp / 111.714640912) / 1.5620688421))
        lee[u] = (100, q, 0.0, q)
    if not lee:
        return lee, None
    best = max(lee, key=lambda u: lee[u][3])
    return lee, (lee[best][3], lee[best][2])


T2_NN_FILES = ["wdl_sf_constants.json", "sf_to_leela.json", "wdl_ext_params_v4.json", "wdl_endgame_p1.py", "wdl_mix_q_classical.json",
               "deepening_params.json"]   # nn/ files Toby 2's Toby 1.0 modules read (Move Evaluations, Endgame Evaluation M, Deepening)


def t2_say_nn_missing():
    """Once: nn/ files the Toby 2 answer reads are missing (a damaged or partial package): said, the answer still comes (their fallbacks)."""
    if T2.get("said_nn"):
        return
    T2["said_nn"] = True
    def present(f):   # a module counts when it can be imported (frozen into toby.exe: no .py in nn/); data files must be there
        if not f.endswith(".py"):
            return (HERE / "nn" / f).exists()
        try:
            import importlib.util
            if str(HERE / "nn") not in sys.path:
                sys.path.append(str(HERE / "nn"))
            return importlib.util.find_spec(f[:-3]) is not None
        except Exception:  # noqa: BLE001
            return False
    miss = [f for f in T2_NN_FILES if not present(f)]
    if _NNC is None:
        miss.append("corrector.py / corrector_weights.json")
    if miss:
        out("info string Toby 2: files missing in nn/: " + ", ".join(miss) + " (Move Evaluations and the position's W/D/L use fallbacks; "
            "reinstall Toby)")


def _search_t2(b, E, opp, t0, emit_, movetime):
    """One Toby 2 search: returns the distribution, or None (Toby 1.0 answers: said, never silent)."""
    net = load_t2()
    if net is not None and T2.get("cfg") is not None and T2["cfg"].get("head"):   # Accurate: the background probe's verdict
        if not t2_probe_settle():
            T2["probe_pending_said"] = True
        elif T2.get("standard_after_probe"):
            T2["standard_after_probe"] = False
            net = load_t2()
    if net is not None and net.device != t2_device():   # Network Device changed while the background load ran: reload on the asked device
        T2.update(net=None, search=None)
        net = load_t2()
    if net is None:
        out(f"info string Toby 2 unavailable ({T2['error']}): {T1_ANSWERS}")
        return None
    Eo = opp[0] if opp else E
    mode = opts["mode"]
    side = t2_side(net)
    values = movetime is None or movetime >= T2_VALUES_BELOW
    src = opts["t2_candidates"]
    if src == "Maia C":
        wait_maia(ft=True)
        if maia_ft is None:
            out("info string Toby 2: the fine-tuned Maia is not loaded, candidates from the network")
            src = "Network"
    S = t2_searcher()
    if S.eng is None:
        out("info string Toby 2: Stockfish not found, network alone (value inputs masked)")
        values = False
    _T2MOD._E_box = (E, Eo)
    want_head = t2_head_wanted()
    pending = T2.get("probe") is not None and T2["probe"].is_alive()
    leela = None if pending else (t2_leela() if want_head else None)
    why = "the GPU / lc0 probe is still running" if pending else T2["leela_error"]
    if want_head and leela is None:
        out(f"info string Toby 2 Accurate: Leela unavailable ({why}): network B with Stockfish values answers (head off)")
    want_t1, want_t2 = t2_eval_wants(movetime)
    if want_t1:   # a Toby 1.0 module (Move Evaluations = Toby 1.0, Position Evaluation = Toby 1.0 / Mix): Maia is loaded now if needed
        wait_maia(timeout=T2_MAIA_WAIT)
    pre = None
    if values and S.eng is not None and maia is not None and T2_EVAL_PREFETCH and t2_prefetch_ok() and want_t1:
        pre = EvalPrefetch()
    _EVPRE[0] = pre
    kids = None
    if want_t2:   # Toby 2's per-move W/D/L: network B on the positions after the moves (batch during the Stockfish complements)
        kids = T2Children(S, b, E, Eo, t2_side(net, child=True), boost=t2_child_threads(net) if pre is None else None)
    if T2_PROVISIONAL and values and S.eng is not None and opts.get("provisional", True) and not getattr(_MUTE, "on", False) \
            and CLOCK is None and not stop_flag.is_set():   # 2.2.1: network B alone at once (stage 1, final 0), then the full answer
        # as before; never for a clocked game move (go wtime / btime: Siby reads only bestmove there, and it would spend the clock)
        t2_provisional(S, b, E, Eo, net, t0, emit_)
    try:
        return _search_t2_answer(S, b, E, Eo, opp, side, src, values, leela, want_head, why, t0, emit_, movetime, mode, pre, kids)
    finally:
        _EVPRE[0] = None


T2_PROVISIONAL = not TOBY1_SHIPPED and os.environ.get("TOBY2_PROVISIONAL", "1") != "0"   # 2.2.1 (no Toby 1.0): the provisional block

# ---------------------------------------------------------------------------------------------- W/D/L correction (product field)
# DRAFT (nn/PROTOCOL_TOBY2_PASS3_DRAFT.md 14, 14.2, 14.3; the reference implementation is nn/sealed_t2_pass3/wdl_y.py, which the offline
# checks import): the product file's "wdl_correction": {"classical": RULE}. Each child W/D/L (the root mover's view) is blended with a
# Stockfish W/D/L built from that move's root value (form "sf_blend_gap", the rule: with the two ratings; "sf_blend", the earlier Y60:
# without); a cadence group that is not in the field = no correction (rapid / blitz). Absent field: the engine is 2.2.1 without it.
Y60_FORMS = ("sf_blend_gap", "sf_blend")
Y60_RAW = {}   # this search's children: {uci: (raw (W, D, L), root value or None, E_child, E_opp)} for deepen (the deepened moves' half-scale view)


def y60_check(cfg):
    """Raises on a malformed wdl_correction field (wdl_y.check_rule: any other form or key set is refused, never a silent guess)."""
    f = (cfg or {}).get("wdl_correction")
    if f is None:
        return
    if not isinstance(f, dict) or not set(f) <= {"classical", "rapid_blitz"}:
        raise ValueError(f"wdl_correction: unknown groups {sorted(f) if isinstance(f, dict) else f}")
    for g, rule in f.items():
        if not isinstance(rule, dict) or rule.get("form") not in Y60_FORMS:
            raise ValueError(f"wdl_correction.{g}: unknown form {rule.get('form') if isinstance(rule, dict) else rule!r}")
        need = {"form", "g0", "tau", "deepened_scale"} | ({"a", "b", "c"} if rule["form"] == "sf_blend_gap" else set())
        if set(rule) != need:
            raise ValueError(f"wdl_correction.{g}: keys {sorted(rule)} != {sorted(need)}")


def y60_spec():
    """The wdl_correction rule of the current cadence (dict), or None (no correction)."""
    f = (T2.get("cfg") or {}).get("wdl_correction")
    if not f or _T2MOD is None:
        return None
    return f.get(_T2MOD.PENALTY_GROUP.get(_T2MOD.TIME_CONTROL_CADENCE[opts["mode"]], ""))


def y60_share(v_m, E_child, E_opp, rule):
    """wdl_y.sf_share: the ROOT mover's share s (0-1) from move m's root value v_m (root mover's view)."""
    e = _T2MOD.expected_score(E_child, -v_m) / 100.0
    if rule["form"] == "sf_blend":
        return 1.0 - e
    e = min(max(e, 1e-6), 1.0 - 1e-6)
    z = float(rule["a"]) * math.log(e / (1.0 - e)) + float(rule["b"]) * (E_child - E_opp) / 400.0 + float(rule["c"])
    return 1.0 - 1.0 / (1.0 + math.exp(-z))


def y60_correct(c, v_m, E_child, E_opp, rule, scale=1.0):
    """wdl_y.correct: c = (W, D, L) of the root mover, v_m = the move's root Stockfish value (cp, mover's view), E_child = the child mover's
    Elo (the root's Eo), E_opp = the root mover's Elo (E); scale = 1, or rule["deepened_scale"] for a deepened move's view."""
    W, D, L = (float(x) for x in c)
    s = y60_share(v_m, E_child, E_opp, rule)
    d = min(D, 2.0 * min(s, 1.0 - s))
    S = (s - d / 2.0, d, 1.0 - s - d / 2.0)
    gap = abs(100.0 * (W + D / 2.0) - 100.0 * (S[0] + S[1] / 2.0))
    z = (gap - float(rule["g0"])) / float(rule["tau"])
    v = scale * (1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0)
    lg = [(1.0 - v) * math.log(max(a, 1e-9)) + v * math.log(max(b, 1e-9)) for a, b in zip((W, D, L), S)]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    t = sum(e)
    return [x / t for x in e]


def t2_provisional(S, b, E, Eo, net, t0, emit_):
    """2.2.1 (Yosha): with Provisional Lines = true, every Toby 2 search shows network B alone (value inputs masked: the path of a move under
    TOBY2_VALUES_BELOW) at once as a provisional block (stage 1, final 0: Siby greys it out and replaces it), then the full answer comes
    as before (stage 3, final 1). The display state the final block reads (ROOT_LAST, LISTED, PRIOR) is restored after this block, so
    the final block is exactly the one 2.2 prints. Kept in T2["prov"]: the move played if a stop comes before the full answer."""
    saved = (list(ROOT_LAST), set(LISTED), dict(PRIOR))
    try:
        r = S.run(b, E, Eo, t2_side(net), source="Network", values=False)
        dist = {u: float(math.exp(lp)) for u, lp in zip(r["legal"], r["logp"])}
        l_, d_, w_ = (float(x) for x in r["wdl"])
        PRIOR.clear()
        PRIOR.update(dist)
        emit_(1, dist, {}, {}, None, 1, t0, True, None, (w_ - l_, d_))
        T2["prov"] = (b.fen(), dist)
    except Exception as exc:  # noqa: BLE001   (display only: the full answer still comes)
        out(f"info string Toby 2: provisional answer unavailable ({type(exc).__name__}: {exc})")
    finally:
        ROOT_LAST[:] = saved[0]
        LISTED.clear(); LISTED.update(saved[1])
        PRIOR.clear(); PRIOR.update(saved[2])


def t2_eval_wants(movetime):
    """(want_t1, want_t2) of a Toby 2 search: whether Toby 1.0's evaluation module runs (Move Evaluations = Toby 1.0 in analysis, or
    Position Evaluation = Toby 1.0 / Mix), and whether Toby 2's child W/D/L run (Move Evaluations = Toby 2 in analysis, or Position
    Evaluation = Toby 2: the position's W/D/L is then the sum over the moves of p x their W/D/L, in play too)."""
    analysis = movetime is None
    mv, pv = opts["t2_move_evals"], opts["t2_position_eval"]
    return (mv == "Toby 1.0" and analysis) or pv in ("Toby 1.0", "Mix"), (mv == "Toby 2" and analysis) or pv == "Toby 2"


T2_CHILD_PREFETCH = os.environ.get("TOBY2_CHILD_PREFETCH", "1") != "0"   # child batch during the complements (0 = after the search)
T2_CHILD_THREADS = int(os.environ.get("TOBY2_CHILD_THREADS", "0"))   # torch threads of the child batch on the processor (0 = automatic)


def t2_child_threads(net):
    """Torch threads for the child W/D/L batch when network B runs on the processor, or None (= the network's own count). During the
    Stockfish complements only Stockfish's two single-thread engines are busy: the batch may use the other usable cores (at most 8).
    The final heads pass of the distribution waits for the batch (Toby2Search.run before_heads) and runs with the network's own count:
    the distribution is the same as without the batch. Off when another torch job may run at the same time (Deepening's body pipelining,
    Toby 1.0's Maia prefetch)."""
    if str(net.device).startswith("cuda"):
        return None
    th = opts["t2_threads"] or (2 if n_cores() >= 4 else 1)
    k = T2_CHILD_THREADS or int(max(th, min(8, usable_cores() - 2)))
    return k if k > th else None


class T2Children:
    """Toby 2's per-move W/D/L for one search (Move Evaluations = Toby 2 / Position Evaluation = Toby 2, 2.0.2). For a human player the
    position's value is V(position) = sum over the moves m of p(m) x V(after m), with p = Toby 2's move probabilities and V(after m) =
    network B's W/D/L of the position after m (toby2.Toby2Net.child_wdl: B's body + W/D/L head, value inputs masked, the child's
    ratings / clocks / history), flipped to the mover's view.
    Moves evaluated (toby2.child_set): the most probable until 99 % of the probability is covered, at least 3, plus the most probable
    move and Stockfish's best move. The probability left over (< 1 %) is spread over the evaluated moves in proportion to their
    probability: V = sum_evaluated p V / sum_evaluated p (the covered mass is printed).
    Timing (as the 2.0.1 Maia prefetch): once Stockfish pass 1 is done (Toby2Search.run on_base), a thread takes B's heads on the pass-1
    values (provisional distribution), chooses its moves by the same rule and runs their batch while the 250k complements run; after
    the search, the final distribution's moves not in that batch get a second batch. The answer's distribution is never changed."""

    def __init__(self, S, b, E, Eo, side_c, boost=None):
        self.S, self.b = S, b.copy()
        self.E, self.Eo, self.side_c = E, Eo, side_c   # E / Eo: the ROOT mover's / opponent's strengths (the child swaps them)
        self.boost = boost if T2_CHILD_PREFETCH else None
        self.base_th = opts["t2_threads"] or (2 if n_cores() >= 4 else 1)
        self.got, self.term, self.th, self.err = {}, {}, None, None
        self.stats = {"prefetched": 0, "missing": 0, "t_prefetch": 0.0, "t_missing": 0.0}
        for m in self.b.legal_moves:
            c = self.b.copy(stack=False)
            c.push(m)
            if c.is_checkmate():
                self.term[m.uci()] = (1.0, 0.0, 0.0)   # (child's loss, draw, win): mated there, the mover wins
            elif c.is_game_over():
                self.term[m.uci()] = (0.0, 1.0, 0.0)

    def _compute(self, moves, boost):
        todo = [u for u in moves if u not in self.got and u not in self.term]
        if not todo:
            return 0
        torch = self.S.net.torch
        if boost:
            torch.set_num_threads(int(boost))
        try:
            self.got.update(self.S.net.child_wdl(self.b, todo, self.Eo, self.E, self.side_c))
        finally:
            if boost:
                torch.set_num_threads(int(self.base_th))
        return len(todo)

    def on_base(self, enc, bout, base, lf_get):
        """Toby2Search.run, pass 1 done (main thread, must not block): the prefetch thread."""
        if not T2_CHILD_PREFETCH:
            return
        boost = self.boost if self.S._prebody is None else None   # Deepening's next body pass is running: no thread change

        def _run():
            try:
                t = time.time()
                f, s = self.S.net.features(enc, base, self.E)
                lp, _ = self.S.net.heads(enc, bout, f, s)
                prov = _T2MOD.child_set(enc["legal"], lp, extra=[max(base, key=base.get)])
                self.stats["prefetched"] = self._compute(prov, boost)
                self.stats["t_prefetch"] = time.time() - t
            except Exception as exc:  # noqa: BLE001   (the moves are then computed after the search)
                self.err = f"{type(exc).__name__}: {exc}"
        self.th = WThread(target=_run, daemon=True)
        self.th.start()

    def wait(self):
        """Toby2Search.run before_heads: the prefetch batch ends before the final heads pass."""
        if self.th is not None:
            self.th.join(T2_EVAL_TIMEOUT)   # bounded: a blocked batch must not hold the search (finish then says so)

    def finish(self, legal, logp, values):
        """-> ({uci: (W - L, D) for the mover}, (W - L, D) of the position = sum p V / covered, info) for the final distribution."""
        self.wait()
        if self.th is not None and self.th.is_alive():
            raise RuntimeError(f"the child batch did not finish within {T2_EVAL_TIMEOUT:.0f} s")
        dist = {u: math.exp(float(v)) for u, v in zip(legal, logp)}
        extra = [max(legal, key=dist.get)] + ([max(values, key=values.get)] if values else [])
        moves = _T2MOD.child_set(legal, logp, extra=extra)
        t = time.time()
        self.stats["missing"] = self._compute(moves, self.boost)
        self.stats["t_missing"] = time.time() - t
        evals, tot, acc = {}, 0.0, [0.0, 0.0, 0.0]
        spec = y60_spec()
        y60 = {}
        Y60_RAW.clear()
        for u in moves:
            lc, dc, wc = self.term.get(u) or self.got[u]
            W, D, L = lc, dc, wc   # the child's side to move is the opponent: flipped to the mover's view
            if spec is not None:   # Y60: corrected before anything reads it (move views, EV_LAST, the position line)
                v_m = values.get(u) if values else None
                Y60_RAW[u] = ((W, D, L), v_m, self.Eo, self.E)
                if v_m is not None:   # a child without a value keeps its raw W/D/L (it is still counted)
                    W, D, L = y60_correct((W, D, L), v_m, self.Eo, self.E, spec)
                y60[u] = [float(W), float(D), float(L)]
            evals[u] = (W - L, D)
            p = dist[u]
            tot += p
            acc[0] += p * W
            acc[1] += p * D
            acc[2] += p * L
        W, D, L = (a / tot for a in acc) if tot > 0 else (0.0, 1.0, 0.0)
        info = dict(self.stats, evaluated=len(moves), covered=tot, moves=moves, err=self.err,
                    wdl={u: [float(x) for x in (self.term.get(u) or self.got[u])] for u in moves})   # the mover's (W, D, L), raw
        if spec is not None:
            info["wdl_y60"] = y60   # the corrected ones (dumps: offline checks recompute them from the raw)
        return evals, (W - L, D), (W, D, L), info


T2_EVAL_PREFETCH = os.environ.get("TOBY2_EVAL_PREFETCH", "1") != "0"   # Move Evaluations prefetch (output-identical; 0 = off, for checks)


def t2_prefetch_ok():
    """The prefetch runs Maia while the Stockfish complements run: on the GPU it costs no core; on the processor only when a core is free
    for it besides those the search already uses (usable_cores, t2_cores_used)."""
    if maia is not None and str(getattr(maia[0].cfg, "device", "cpu")).startswith("cuda"):
        return True
    return usable_cores() - t2_cores_used() - (1 if T2.get("spec3") is not None else 0) >= T2_PREFETCH_CPU_FREE


T2_PREFETCH_CPU_FREE = float(os.environ.get("TOBY2_PREFETCH_CPU_FREE", "1000"))   # free cores needed when Maia runs on the processor: off by default
# (CPU pod: median -0.10 to -0.23 s but p95 +0.4 s, a missed prefetch holds Maia for ~0.35 s; TOBY2_PREFETCH_CPU_FREE=2 turns it on)


def _evals_prefetch_fn(S, b, E, x, opp, legal_box):
    """The work of an EvalPrefetch (thread), from the pass-1 values: see EvalPrefetch."""
    oppx = (opp[0], opp[1]) if opp else None

    def fn(pre):
        enc, bout, base, lf_get = legal_box
        lf = lf_get()
        f, s = S.net.features(enc, base, E)
        lp, _ = S.net.heads(enc, bout, f, s, lf)
        dist = {u: float(math.exp(v)) for u, v in zip(enc["legal"], lp)}
        shown = [u for u, p in sorted(dist.items(), key=lambda kv: -kv[1]) if p >= 0.001][:SF_EVAL_MOVES]
        consts = None if opp else wdl_sf_consts()
        ev_elo, ev_opp, we, qp = evals_setup(E, x, oppx, consts, "sf")
        after, elo, oppo, allpol = evals_batch(b, shown, we, qp, ev_elo, ev_opp, oppx)
        todo, pol, need = _maia_values_plan(after, elo, oppo, allpol)
        if todo:
            pre.key = _maia_values_key(todo, elo, oppo, pol, need)
        if not mix_on() and not (we and we[0].applies(b.fen())) and (b.fen(), ev_elo, ev_opp) not in _MV:
            pre.single_key = (_board_id(b), ev_elo, ev_opp)
        pre.planned.set()
        if pre.key is not None:
            pre.batch = _maia_values_run(todo, elo, oppo, pol, need)
        pre.done.set()
        if pre.single_key is not None:
            pre.single = maia_value(b, ev_elo, ev_opp)
        pre.single_done.set()
    return fn


T2_EVAL_TIMEOUT = float(os.environ.get("TOBY2_EVAL_TIMEOUT", "30"))   # seconds: an evaluation module that does not answer is turned off
T2_MAIA_WAIT = float(os.environ.get("TOBY2_MAIA_WAIT", "120"))   # seconds a Toby 2 search waits for Maia (Toby 1.0 modules only)


def _bounded(fn, seconds):
    """Runs fn() in a daemon thread: (True, result), or (False, reason) when it raised or did not finish within `seconds` (a missing
    module or file may block or fail: the search must still answer and send its bestmove)."""
    box = {}
    tl = (getattr(_MUTE, "on", False), getattr(_MUTE, "capture", None), getattr(_FT, "force", None))   # this thread's state goes along

    def _run():
        _MUTE.on, _MUTE.capture, _FT.force = tl
        try:
            box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["e"] = f"{type(exc).__name__}: {exc}"
    th = WThread(target=_run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        return False, f"no answer within {seconds:.0f} s"
    if "e" in box:
        return False, box["e"]
    return True, box["v"]


def _t1_evals(b, E, x, dist, lee, root, opp, net_root):
    """Toby 1.0's evaluation module on a Toby 2 answer (Move Evaluations = Toby 1.0, Position Evaluation = Toby 1.0 / Mix): (evals, root_eval)."""
    evals, root_eval = None, net_root
    shown = [u for u, p in sorted(dist.items(), key=lambda kv: -kv[1]) if p >= 0.001][:SF_EVAL_MOVES]
    consts = None if opp else wdl_sf_consts()
    oppx = (opp[0], opp[1]) if opp else None
    if consts:
        evals, rev = toby_evals(b, E, x, shown, lee, {}, root, time.time() + 12, oppx, consts=consts, src="sf", dist=dist)
    else:
        lee_w = {u: (v[0], sf_to_leela_q(v[1]), 0.0, sf_to_leela_q(v[3])) for u, v in lee.items()}
        evals, rev = toby_evals(b, E, x, shown, lee_w, {}, (sf_to_leela_q(root[0]), 0.0), time.time() + 12, oppx, src="sf", dist=dist)
    if opts["t2_position_eval"] == "Toby 1.0":
        root_eval = rev
    elif opts["t2_position_eval"] == "Mix" and rev:
        to_wdl = lambda wl, d: ((1 - d + wl) / 2, d, (1 - d - wl) / 2)
        a1, a2 = to_wdl(*net_root), to_wdl(*rev)
        m = [(1 - T2_MIX_W) * p + T2_MIX_W * q for p, q in zip(a1, a2)]
        root_eval = (m[0] - m[2], m[1])
    if opts["t2_move_evals"] != "Toby 1.0":
        evals = None
    return evals, root_eval


def _search_t2_answer(S, b, E, Eo, opp, side, src, values, leela, want_head, why, t0, emit_, movetime, mode, pre, kids=None):
    on_base = None
    if pre is not None:
        x0 = rule_params(E)[0]

        def on_base(enc, bout, base, lf_get):   # called by Toby2Search.run once pass 1 is done (main thread): starts the prefetch thread
            pre.start(_evals_prefetch_fn(S, b, E, x0, opp, (enc, bout, base, lf_get)))
    if kids is not None:   # Toby 2's child W/D/L batch also starts once pass 1 is done (after Toby 1.0's prefetch when both run)
        on_base0 = on_base

        def on_base(enc, bout, base, lf_get):
            if on_base0 is not None:
                on_base0(enc, bout, base, lf_get)
            kids.on_base(enc, bout, base, lf_get)
    try:
        r = S.run(b, E, Eo, side, source=src, values=values, leela=leela, on_base=on_base,
                  before_heads=kids.wait if kids is not None else None)
    except Exception as exc:  # noqa: BLE001
        out(f"info string Toby 2 failed ({type(exc).__name__}: {exc}): {T1_ANSWERS}")
        T2["fb_said"] = True   # 2.2.1 fallback: the failure is already said
        return None
    if r is None or stop_flag.is_set():
        return None
    cfg = T2.get("cfg")
    if cfg and cfg.get("blunder_penalty") and r.get("values"):   # product-file field (toby2_blunder_fix.md); absent: nothing changes
        cad = _T2MOD.TIME_CONTROL_CADENCE[mode]
        r["logp"], beta = _T2MOD.blunder_penalty(r["logp"], r["legal"], r["values"], E, cad, cfg["blunder_penalty"])
        r["path"]["blunder_penalty"] = f"beta {beta:.4f}" if beta > 0 else "off for this cadence / rating"
    if T2_SF_THREADS != 1:
        r["path"]["sf_threads"] = f"{T2_SF_THREADS} (NOT the values recipe: development only)"
    if T2.get("accurate_unavailable"):
        r["path"].update(requested="Toby 2 Accurate", accurate_unavailable=T2["accurate_unavailable"])
    r["path"].update(product=cfg["product"] if cfg else "none (no product file)", config_sha256=cfg["config_sha256"][:16] if cfg else "-",
                     cadence=_T2MOD.TIME_CONTROL_CADENCE[mode], clock="game clocks" if CLOCK is not None else "cadence only")
    if want_head and leela is None:
        r["path"].update(head="off", leela=f"unavailable: {why}")
    elif not want_head:
        r["path"].setdefault("head", "off" if (cfg and cfg.get("head")) else "-")
    if "leela" in r["path"] and str(r["path"]["leela"]).startswith("failed"):
        out(f"info string Toby 2 Accurate: Leela search {r['path']['leela']}: network B with Stockfish values answers (head off)")
        t2_leela_failed(r["path"]["leela"])
    l_, d_, w_ = (float(x) for x in r["wdl"])
    net_root = (w_ - l_, d_)
    ev2 = rev2 = kinfo = None
    if kids is not None and r["values"] is not None and not stop_flag.is_set():   # Toby 2's per-move W/D/L (2.0.2)
        tk = time.time()
        ok, got = _bounded(lambda: kids.finish(r["legal"], r["logp"], r["values"]), T2_EVAL_TIMEOUT)
        if ok:
            ev2, rev2, _, kinfo = got
            kinfo["t_finish"] = time.time() - tk
        else:
            out(f"info string Toby 2: per-move W/D/L unavailable ({got}): Move Evaluations Off for this move, the position's W/D/L from "
                "network B's head")
    if os.environ.get("TOBY2_DUMP"):   # development / parity: the exact answer, one JSON line per search
        try:
            d2 = {"fen": b.fen(), "moves": [m.uci() for m in b.move_stack], "E": E, "Eo": Eo, "legal": r["legal"],
                  "logp": [float(x) for x in r["logp"]], "wdl": [float(x) for x in r["wdl"]], "path": r["path"],
                  "values": r["values"], "candidates": r["candidates"], "timings": r["timings"],
                  "lfeat": None if r.get("lfeat") is None else [[float(x) for x in row] for row in r["lfeat"]]}
            if kinfo is not None:   # 2.0.2: Toby 2's child W/D/L (the mover's W, D, L per evaluated move) and the position's sum
                d2["evals2"] = {"moves": kinfo["moves"], "wdl": kinfo["wdl"], "covered": kinfo["covered"], "wdl_y60": kinfo.get("wdl_y60"),
                                "position": [(1 - rev2[1] + rev2[0]) / 2, rev2[1], (1 - rev2[1] - rev2[0]) / 2]}
            with open(os.environ["TOBY2_DUMP"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(d2) + "\n")
        except Exception as exc:  # noqa: BLE001
            out(f"info string Toby 2: dump failed ({exc})")
    dist = {u: float(math.exp(lp)) for u, lp in zip(r["legal"], r["logp"])}
    tm = r["timings"]
    out("info string toby path " + json.dumps(r["path"], separators=(",", ":")))
    out("info string toby timing toby2 " + " ".join(f"{k} {v:.3f}s" for k, v in tm.items() if isinstance(v, float)))
    PRIOR.clear()
    PRIOR.update({u: float(math.exp(lp)) for u, lp in zip(r["legal"], r.get("masked_logp", r["logp"]))})
    sf = r["values"]
    if sf is None:   # network alone: no Stockfish line; the position's W/D/L from the network
        if kids is not None:
            out("info string toby evaluation position head (network alone, no Stockfish values: no per-move W/D/L)")
        emit_(3, dist, {}, {}, None, 1, t0, False, None, net_root)
        return dist
    try:
        lee, root = _sf_pseudo_leela(sf)
    except Exception as exc:  # noqa: BLE001   (display only: the lines then carry no Stockfish-derived value)
        out(f"info string Toby 2: Stockfish values not converted ({type(exc).__name__}: {exc})")
        lee, root = {}, None
    x = rule_params(E)[0]
    if opts["move_choice"] == "human" and opts["t2_blunder_check"] and mode == "Classical" and _NNC is not None:   # Human mode: Toby 1.0's play-time check
        es = {u: _NNC.expected_score(E, cp) if abs(cp) < 10000 else (1.0 if cp > 0 else 0.0) for u, cp in sf.items()}
        top = max(es.values())
        PLAY_LOSS.update(E=E, loss={u: 100 * (top - v) for u, v in es.items()})
    evals, root_eval = None, net_root
    want_t1, _ = t2_eval_wants(movetime)
    if want_t1 and maia is None:
        out(f"info string Toby 2: Toby 1.0's evaluation module unavailable (Maia not loaded: {load_error or ('still loading' if not ready.is_set() else 'not in this package')}): "
            "its Move / Position Evaluations are Off for this move")
    if want_t1 and maia is not None and root is not None and not stop_flag.is_set():
        ok, got = _bounded(lambda: _t1_evals(b, E, x, dist, lee, root, opp, net_root), T2_EVAL_TIMEOUT)
        if ok:
            evals, root_eval = got
        else:
            out(f"info string Toby 2: Toby 1.0's evaluation module unavailable ({got}): its Move / Position Evaluations are Off for this move")
        if pre is not None:   # the prefetch's use (timing only: the numbers are the same either way)
            out(f"info string toby timing toby2 evals prefetch_batch {pre.stats['batch_hit']:.0f} prefetch_single {pre.stats['single_hit']:.0f}")
    if ev2 is not None:
        if opts["t2_move_evals"] == "Toby 2" and movetime is None:
            evals = ev2
            T2_SHOWN.update(ev2)   # every evaluated move is listed (also below 0.1 %)
        if opts["t2_position_eval"] == "Toby 2":
            root_eval = rev2
        k_ = kinfo
        out(f"info string toby timing toby2 children {k_['t_prefetch'] + k_['t_finish']:.3f}s prefetched {k_['prefetched']} "
            f"after {k_['missing']} wait {tm.get('children_wait', 0.0):.3f}s" + (f" prefetch_error {k_['err']}" if k_["err"] else ""))
        wh, dh = net_root
        hw, hl = (1 - dh + wh) / 2, (1 - dh - wh) / 2
        sw, sl = (1 - rev2[1] + rev2[0]) / 2, (1 - rev2[1] - rev2[0]) / 2
        out(f"info string toby evaluation moves {opts['t2_move_evals']} position {opts['t2_position_eval']} evaluated {k_['evaluated']} "
            f"covered {k_['covered']:.4f} sum wdl {int(round(1000 * sw))} {int(round(1000 * rev2[1]))} {int(round(1000 * sl))} "
            f"es {sw + rev2[1] / 2:.4f} head wdl {int(round(1000 * hw))} {int(round(1000 * dh))} {int(round(1000 * hl))} "
            f"es {hw + dh / 2:.4f} gap {(sw + rev2[1] / 2) - (hw + dh / 2):+.4f}")
    if stop_flag.is_set():
        return dist
    emit_(3, dist, lee, {}, root, V2_BASE, t0, False, evals, root_eval)
    return dist


def _await_stop(infinite):
    while infinite and not stop_flag.is_set():
        time.sleep(0.02)


def t2_fallback_answer(b, legal, t0, infinite=False):
    """2.2.1 (no Toby 1.0 shipped): the answer when Toby 2's full answer is not there: the time ran out or a 'stop' came first, or Toby 2
    failed. Network B alone, value inputs masked (the same path as a move with less than TOBY2_VALUES_BELOW seconds: no Stockfish values,
    so no blunder penalty), said in an info string; if network B is unavailable, Stockfish's move at the base budget, else the first legal
    move, said too. Never silent, never Maia."""
    prov = T2.pop("prov", None)
    if prov is not None and prov[0] == b.fen() and stop_flag.is_set():   # 2.2.1: the provisional block was shown: its move, no new block
        out("info string Toby 2: stopped before the full answer: the provisional answer (network B alone) is the move")
        _await_stop(infinite)   # go infinite: bestmove only after the GUI's stop (UCI)
        out(f"bestmove {pick_move(prov[1], b)}")
        return
    said = T2.pop("fb_said", False) or T2.get("error") or not t2_on()   # a failure already said (Toby 2 failed / unavailable / not found)
    try:
        net = load_t2()
        if net is None:
            raise RuntimeError(T2.get("error") or "network B unavailable")
        E, opp = side_ratings(b)
        S = t2_searcher()
        r = S.run(b, E, opp[0] if opp else E, t2_side(net), source="Network", values=False)
        dist = {u: float(math.exp(lp)) for u, lp in zip(r["legal"], r["logp"])}
        if stop_flag.is_set():
            msg = "the time ran out before the full answer: network B alone"
        elif said:
            msg = "network B alone"
        else:   # no stop and no error: Stockfish's search ended without a value for every legal move (StockfishCalc.evaluate -> None)
            msg = "Stockfish gave no complete set of values: network B alone"
        out(f"info string Toby 2: {msg} answers (value inputs masked, no blunder penalty) ({time.time() - t0:.2f}s)")
        out("info string toby path " + json.dumps(dict(r["path"], fallback="network alone"), separators=(",", ":")))
        _await_stop(infinite)   # go infinite: bestmove only after the GUI's stop (UCI)
        out(f"bestmove {pick_move(dist, b)}")
        return
    except Exception as exc:  # noqa: BLE001
        why2 = f"{type(exc).__name__}: {exc}"
    mv = legal[0]
    try:
        if os.path.exists(SF_PATH):
            e = T2.get("eng") or StockfishCalc(fixed_threads=1)
            v = e.evaluate(b, V2_BASE)
            if v:
                mv = max(v, key=v.get)
    except Exception:  # noqa: BLE001
        pass
    out(f"info string Toby 2 unavailable ({why2}): Stockfish's move answers (no Toby 1.0 in this Toby)")
    _await_stop(infinite)   # go infinite: bestmove only after the GUI's stop (UCI)
    out(f"bestmove {mv}")


def _search(movetime, infinite=False, bd=None, nested=False):
    global lc0, LEELA_BROKEN
    t0 = time.time()
    T2.pop("fb_said", None)
    if not nested:
        T2.pop("prov", None)
    b = (bd if bd is not None else board).copy()
    legal = [m.uci() for m in b.legal_moves]
    if not legal:
        out("bestmove 0000")
        return
    if maia_wanted():   # a Toby 2 search with the default options never waits for Maia (2.0.2: not even loaded)
        wait_maia()
    t2_say_missing()
    if t2_on():
        t2_say_nn_missing()
    if t2_on() and not stop_flag.is_set():   # Toby 2: network B + Toby 1.0's Stockfish recipe (Toby 1.0 answers only if it fails, and says so)
        E, opp = side_ratings(b)
        out(f"info string Toby 2: rating {own_rating(b)} ({opts['source']}) -> strength {strength_text(E)}; time control {opts['mode']}; "
            f"candidates {opts['t2_candidates']}" + (f"; opponent strength {strength_text(opp[0])}" if opp else "; opponent: same rating") + versions_text())
        EV_LAST.clear(); PLAY_LOSS.clear(); ROOT_LAST.clear(); LISTED.clear(); T2_SHOWN.clear()
        EVAL_K[0] = SF_EVAL_MOVES
        d2 = _search_t2(b, E, opp, t0, lambda *a, **k: emit(*a, bd=b, **k), movetime)
        if d2 is not None:
            deep = infinite and opts.get("deepening") and not nested
            if deep and not stop_flag.is_set():
                try:
                    deepen(b, d2, t0)
                except Exception as exc:  # noqa: BLE001
                    out(f"info string toby deepening stopped ({exc})")
                if T2.get("search") is not None:   # no body pass queued past the deepening (a stopped or failed nested search)
                    T2["search"].next_body = None
            if infinite:
                while not stop_flag.is_set():
                    time.sleep(0.02)
            out(f"bestmove {pick_move(d2, b)}")
            return
    if not TOBY1_SHIPPED:   # 2.2.1: no Toby 1.0 to fall back on: Toby 2's network alone (or Stockfish), always said
        t2_fallback_answer(b, legal, t0, infinite and not nested)
        return
    wait_maia()   # Toby 1.0 answers (chosen, or Toby 2 failed / unavailable: said above): Maia is loaded now if it was not
    if maia is None:
        out(f"info string ERROR {load_error}")
        out(f"bestmove {legal[0]}")
        return
    wait_maia_ft()
    E, opp = side_ratings(b)
    x, lam, gam = rule_params(E)
    emit_ = lambda *a, **k: emit(*a, bd=b, **k)   # the board of THIS search (the global one may already hold the next position)
    opp_txt = f"; opponent strength {strength_text(opp[0])} (Maia input {opp[1]})" if opp else "; opponent: same rating"
    out(f"info string Toby: rating {own_rating(b)} ({opts['source']}) -> strength {strength_text(E)}; time control {opts['mode']}; precision {'Fast' if (opts.get('hybrid') or opts['calculator'] == 'Stockfish') else 'Accurate' if opts['calculator'] == 'Auto' else 'forced'}{' (hybrid)' if hybrid_on() else ''} (calculator {calculator()}); "
        f"Maia input rating {x}" + (" (fine-tuned Maia)" if ft_on() else "") + "; " + (("move model: history network" if hist_on() else "move model: neural corrector") if (network_on() or hist_on()) else f"severity lambda = {lam:.2f}, bonus gamma = {gam:.2f}") + opp_txt + versions_text())
    sf_mode = calculator() == "Stockfish" and CORRECTOR_SF is not None
    hmode = hist_on() and not sf_mode
    pm = maia_policy(b, x, opp[1] if opp else None, history=True if (sf_mode or hmode) else None)   # the Stockfish network was trained with Maia given the previous positions
    PRIOR.clear()
    PRIOR.update(pm)
    EV_LAST.clear()
    T2_SHOWN.clear()
    PLAY_LOSS.clear()
    ROOT_LAST.clear()
    LISTED.clear()
    EVAL_K[0] = 30 if calculator() == "Leela" else SF_EVAL_MOVES   # the moves this search evaluates (W/D/L): the only ones listed
    dist = dict(pm)
    prov_on = opts.get("provisional", True)   # option Provisional Lines: false = only the final block (game annotation); the final result is the same
    if prov_on:
        emit_(1, dist, {}, {}, None, 1, t0, True)  # step 1: Maia alone (nodes 1: a GUI may ignore a line with nodes 0)
    if movetime is not None and movetime < MAIA_ONLY_BELOW:   # deep time trouble: Maia alone, at once
        out("info string Toby: little time left, Maia alone")
        emit_(1, dist, {}, {}, None, 1, t0, False)   # Maia's answer is the final one: always shown, also with Provisional Lines = false
        out(f"bestmove {pick_move(dist, b)}")
        return
    best_dist = dist
    if sf_mode and not stop_flag.is_set():   # a single legal move is evaluated like any other (amendment 15): its move view, and the position = that move
        best_dist = _search_sf(b, pm, E, x, opp, t0, emit_) or best_dist
    elif not stop_flag.is_set():
        try:
            ensure_lc0()
            fen = b.fen()
            # step 2 (about 0.5 s): quick provisional answer from the network's static value of every plausible move; every plausible move has a value, none escapes the penalty
            plausible = [u for u, p in sorted(pm.items(), key=lambda kv: -kv[1]) if p >= 0.005]
            qs = lc0.static_values(b, plausible) if prov_on else {}   # used only by the provisional displays, never by the final result
            sv = dict(qs)   # static values known so far

            def interim(dist, lee_, extra_, root_, with_root=True):
                """Toby's evaluation of the 12 most probable moves with the Leela values available at this stage (static, then search, then complements). Maia's values are cached."""
                sh = [u for u, p in sorted(dist.items(), key=lambda kv: -kv[1]) if p >= 0.001][:12]
                missing = [u for u in sh if u not in sv and not (lee_.get(u, (0,))[0] >= 1 or u in extra_)]
                if missing:
                    sv.update(lc0.static_values(b, missing))
                ev_, rev_ = toby_evals(b, E, x, sh, lee_, {**sv, **extra_}, root_, time.time() + 12, opp, dist=dist)
                return ev_, (rev_ if with_root else None)
            if qs and not stop_flag.is_set():
                d2 = toby_distribution_static(pm, qs, E)
                ev2, _ = interim(d2, {}, {}, None, with_root=False)
                emit_(2, d2, {u: (1, q, 0.0, q) for u, q in qs.items()}, {}, None, len(qs), t0, True, ev2, None)
                best_dist = d2
            if not stop_flag.is_set():
                sfc = {}   # History model: Stockfish's values of every move, computed on the CPU while Leela searches on the GPU
                sfraw = {}   # the V2 values in centipawns (extra inputs of the Accurate network T10)
                v2acc = {}   # details of the V2 search (v2info: section-10 inputs of an Accurate network trained with them)
                exp_acc = ft_on() and CORRECTOR_HSF_FT is not None
                sf_thread = None
                if hmode and CORRECTOR_HSF is not None and os.path.exists(SF_PATH):
                    def _sf_values():
                        global sf_engine
                        try:
                            if sf_engine is None:
                                sf_engine = StockfishCalc()
                            r = sf_v2(b, pm, spec=spec_engine(), info=v2acc) if exp_acc else sf_engine.evaluate(b, SF_TRAIN_NODES)
                            if r:
                                sfc.update(_NNC.sf_pseudo_leela(r)[0])
                                if exp_acc:
                                    sfraw.update(r)
                        except Exception:  # noqa: BLE001
                            pass
                    sf_thread = WThread(target=_sf_values, daemon=True)
                    sf_thread.start()
                lee, root = lc0.search(fen, 1024)  # step 3: the main search
                pol = dict(getattr(lc0, "last_policy", {}))   # Leela's prior of the main search (an input of the network; the complements overwrite it)
                pol1 = None
                if CORRECTOR_ACC_MIX is not None and exp_acc:   # ensemble AX: the hybrid network's input, one 1-node evaluation (as the Fast path)
                    try:
                        lc0.search(fen, 1)
                        pol1 = {u: round(float(v), 5) for u, v in dict(getattr(lc0, "last_policy", {})).items()}
                    except Exception:  # noqa: BLE001
                        pol1 = None
                if sf_thread is not None:
                    sf_thread.join(timeout=30)
                cor = None
                if hmode:
                    cor = ((CORRECTOR_HSF_FT if ft_on() else CORRECTOR_HSF) if sfc else CORRECTOR_H)   # Stockfish complements when Stockfish answered, else the 64-node Leela ones
                    if cor is CORRECTOR_HSF_FT and not sfraw:   # T10 needs the V2 values: without them, the 64-node Leela complements
                        cor = CORRECTOR_H
                sfs = sfraw if (exp_acc and cor is CORRECTOR_HSF_FT) else None
                pm2 = (maia_policy(b, own_maia_rating(b), history=True) if cor is not None and cor.meta.get("use_pm2")
                       else maia_policy(b, own_maia_rating(b)) if network_on() and CORRECTOR.meta.get("use_pm2") else None)
                if stop_flag.is_set():  # stopped while it ran: keep the last complete answer (never a distribution built from missing values)
                    out(f"bestmove {pick_move(best_dist, b)}")
                    return
                need = [u for u in plausible if lee.get(u, (0,))[0] < MIN_N]
                extra = {}  # complementary 64-node values: the validated rule, and the only ones used for the final answer
                if cor is not None and cor in (CORRECTOR_HSF, CORRECTOR_HSF_FT):   # complements from Stockfish (Leela scale), no 64-node searches
                    extra = {u: sf_to_leela_q(sfc[u][3]) for u in need if u in sfc}
                    out(f"info string Toby: complements from Stockfish ({len(extra)} moves), no extra Leela searches")
                    need = []
                # provisional: moves not yet complemented keep their static value, so none is favoured while waiting
                d3, _, _ = move_distribution(b, pm, lee, {u: qs[u] for u in need if u in qs}, E, x, opp, root, pol, pm2, cor=cor, sf_shallow=sfs, v2info=(v2acc.get("v2info") if sfs is not None else None))
                if sfs is not None:
                    d3 = acc_mix(d3, b, pm, E, x, opp, sfs, v2acc.get("v2info"), pol1)
                if prov_on:
                    ev3, rev3 = interim(d3, lee, {u: qs[u] for u in need if u in qs}, root)
                    emit_(3, d3, lee, {u: qs[u] for u in need if u in qs}, root, 1024, t0, True, ev3, rev3)
                best_dist = d3
                for k, u in enumerate(need, 1):
                    if stop_flag.is_set():
                        break
                    if prov_on:
                        out(f"info string Toby: checking the moves Leela did not explore, {k}/{len(need)}")
                    mv, _ = lc0.search(fen, 64, searchmoves=u)  # no time limit: the rule needs every complement
                    if stop_flag.is_set():  # interrupted complement: its partial value is not used
                        break
                    if u in mv and mv[u][0] >= 1:
                        extra[u] = mv[u][3]
                    prov = {**{v: qs[v] for v in need if v in qs}, **extra}
                    d3, _, _ = move_distribution(b, pm, lee, prov, E, x, opp, root, pol, pm2, cor=cor, sf_shallow=sfs, v2info=(v2acc.get("v2info") if sfs is not None else None))
                    if sfs is not None:
                        d3 = acc_mix(d3, b, pm, E, x, opp, sfs, v2acc.get("v2info"), pol1)
                    best_dist = d3
                    if k < len(need) and prov_on:
                        ev3, rev3 = interim(d3, lee, prov, root)
                        emit_(3, d3, lee, prov, root, 1024, t0, True, ev3, rev3)
                if stop_flag.is_set():
                    out(f"bestmove {pick_move(best_dist, b)}")
                    return
                d3, _, _ = move_distribution(b, pm, lee, extra, E, x, opp, root, pol, pm2, cor=cor, sf_shallow=sfs, v2info=(v2acc.get("v2info") if sfs is not None else None))
                if sfs is not None:
                    d3 = acc_mix(d3, b, pm, E, x, opp, sfs, v2acc.get("v2info"), pol1)  # final: the network (or exactly the validated rule)
                if prov_on:
                    ev3, rev3 = interim(d3, lee, extra, root)
                    emit_(3, d3, lee, extra, root, 1024, t0, True, ev3, rev3)  # full rule shown first, with the evaluation of the main moves; the complete evaluation follows
                best_dist = d3
                shown = [u for u, p in sorted(d3.items(), key=lambda kv: -kv[1]) if p >= 0.001][:30]
                # a shown move that Leela never valued gets its static value (1 node, about 0.03 s) for the evaluation; never the value of the position itself
                unvalued = [u for u in shown if not ((lee.get(u, (0,))[0] >= 1) or (u in extra))]
                stat = lc0.static_values(b, unvalued) if unvalued else {}
                evals, root_eval = toby_evals(b, E, x, shown, lee, {**stat, **extra}, root, time.time() + 12, opp, dist=d3)
                emit_(3, d3, lee, extra, root, 1024, t0, False, evals, root_eval)
        except Exception as exc:  # noqa: BLE001
            LEELA_BROKEN = True   # Auto / Accurate / Hybrid use Stockfish from now on (lc0 failed on this machine)
            out(f"info string Leela unavailable ({exc}): Maia only for this move, Stockfish from the next one")
            stop_all_lc0()
            lc0 = None
    deep = infinite and opts.get("deepening") and not nested
    if not stop_flag.is_set() and maia is not None and not nested and not deep:   # idle time: Maia's line of the positions that can follow, ready before the user steps forward
        PREFETCH_ABORT.clear()
        threading.Thread(target=prefetch_maia, args=(b, best_dist), daemon=True).start()
    if deep and not stop_flag.is_set():   # option Deepening: the next positions' own evaluations, then one updated final block
        try:
            deepen(b, best_dist, t0)
        except Exception as exc:  # noqa: BLE001
            out(f"info string toby deepening stopped ({exc})")
    if infinite:  # go infinite: keep the display until the stop command, like a classic engine
        while not stop_flag.is_set():
            time.sleep(0.02)
    out(f"bestmove {pick_move(best_dist, b)}")


def _r6(d):
    """Maia's probabilities rounded to 6 decimals, as stored in the training data (nn/compute_maia.py): rounding can tie weak moves, and the
    tie decides their rank, an input of the network."""
    return None if d is None else {u: round(v, 6) for u, v in d.items()}


def _search_sf(b, pm, E, x, opp, t0, emit_):
    """Calculator = Stockfish: one MultiPV search over every legal move, then the Stockfish network and Toby's evaluation. Returns the final
    distribution, or None (stopped, or Stockfish unavailable: Maia's distribution stays)."""
    global sf_engine, LEELA_BROKEN, lc0
    try:
        if sf_engine is None:
            sf_engine = StockfishCalc()
        nodes = opts["sf_nodes"] or sf_nodes_auto(E)
        cor = sf_corrector()
        exp = hybrid_on()
        if hybrid_wanted() and not exp:
            say_fallback("lc0 failed on this machine" if LEELA_BROKEN else ("no NVIDIA GPU detected" if leela_no_driver() else "lc0 or its network not found"))
        pol = {}
        v2 = exp and opts["mode"] == "Classical"   # V2 values in Classical only (the online hybrid was trained on 300k over every move)
        if exp:   # Precision Fast, hybrid path: Leela's raw policy; Classical: V2 values, network H8 (or its successor); Rapid / Blitz: online network
            try:
                pol = leela_raw_policy(b)
                cor = hybrid_net()
            except Exception as lexc:  # noqa: BLE001   (lc0 failed: this search and the next ones are Fast)
                LEELA_BROKEN = True
                stop_all_lc0()
                lc0 = None
                exp, v2, pol = False, False, {}
                say_fallback(lexc)
        if v2:
            out(f"info string Toby: Fast (Stockfish {V2_BASE // 1000}k + {V2_EXT // 1000}k on the candidates, Leela's raw policy)")
        elif exp:
            out(f"info string Toby: Fast (Stockfish {nodes // 1000}k, Leela's raw policy)")
        pm2 = maia_policy(b, own_maia_rating(b), history=True) if cor.meta.get("use_pm2") else None
        v2i = {}
        sf = sf_v2(b, pm, spec=spec_engine(), info=v2i) if v2 else sf_engine.evaluate(b, nodes)
        if sf is None:
            return None
        lee, root = _NNC.sf_pseudo_leela(sf)
        moves, ctx, mv, base = _NNC.build_inputs(b, _r6(pm), lee, pol, {}, root, x, E, opts["mode"].lower(), venue(),
                                                 E_opp=(opp[0] if opp else None), pm2=_r6(pm2), history=True, v2info=v2i.get("v2info"),
                                                 sf_budget=budget_input(cor, nodes))
        dist = cor.distribution(ctx, mv, base, moves)
        if v2 and CORRECTOR_HYB_MIX is not None and cor is CORRECTOR_HYB:   # ensemble HF: mixed log-probabilities, renormalised
            d2 = CORRECTOR_HYB_MIX.distribution(ctx, mv, base, moves)
            w = HYB_MIX_W
            z = {u: math.exp((1 - w) * math.log(max(dist.get(u, 0.0), 1e-12)) + w * math.log(max(d2.get(u, 0.0), 1e-12))) for u in dist}
            t = sum(z.values())
            dist = {u: v / t for u, v in z.items()}
        if opts["move_choice"] == "human" and not exp and opts["mode"] == "Classical" and sf:   # play realism (see PLAY_LAMBDA)
            es = {u: _NNC.expected_score(E, cp) if abs(cp) < 10000 else (1.0 if cp > 0 else 0.0) for u, cp in sf.items()}
            top = max(es.values())
            PLAY_LOSS.update(E=E, loss={u: 100 * (top - v) for u, v in es.items()})
        shown = [u for u, p in sorted(dist.items(), key=lambda kv: -kv[1]) if p >= 0.001][:SF_EVAL_MOVES]
        consts = None if opp else wdl_sf_consts()
        if opts["move_choice"] == "human" and os.environ.get("TOBY_PLAY_EVALS", "1") == "0":   # (measure) a game move: no evaluation of each move
            evals, root_eval = None, None
        elif consts:   # constants fitted on Stockfish's values: they read them directly
            evals, root_eval = toby_evals(b, E, x, shown, lee, {}, root, time.time() + 12, opp, consts=consts, src="sf", dist=dist)
        else:        # opponent's rating given (no Stockfish fit yet): Leela-scaled values with the Leela constants
            lee_w = {u: (v[0], sf_to_leela_q(v[1]), 0.0, sf_to_leela_q(v[3])) for u, v in lee.items()}
            root_w = (sf_to_leela_q(root[0]), 0.0)
            evals, root_eval = toby_evals(b, E, x, shown, lee_w, {}, root_w, time.time() + 12, opp, src="sf", dist=dist)
        if stop_flag.is_set():
            return dist
        emit_(3, dist, lee, {}, root, nodes, t0, False, evals, root_eval)   # MultiPV scores: Stockfish's own centipawns
        return dist
    except Exception as exc:  # noqa: BLE001
        out(f"info string Stockfish unavailable ({exc}): Maia only")
        stop_stockfish()
        return None


# ----------------------------------------------------------------------------------------------------- likelihood (Elo estimate in Siby)
def maia_own_rating(r):
    """Maia rating of a player of rating r in the current Rating Source (as own_maia_rating does for the options)."""
    if ft_on():
        return rule_params(strength_E(r, opts["source"]))[0]
    if opts["source"].startswith("Chess.com"):
        r = from_fide(strength_E(r, opts["source"]), "lichess", opts["mode"].lower() if opts["mode"] != "Classical" else "classical")
    return int(min(max(r, 600), 3500))


def _score_txt(cp, mate):
    return f"mate {mate}" if mate is not None else f"cp {cp}"


# Network of 'toby likelihood' in Classical (option Likelihood Model; the method page). T5 (default): the Fast path of sealed
# test v5 (Stockfish 300k over every move, network T5), the one Siby's level estimate was calibrated with on 3 November. H8 and H6b read the
# V2 Stockfish values (150k over every move + 250k on each candidate set) and Leela's raw policy, as the hybrid search path does; H8s is the
# network H8 fed with the same Stockfish 300k search as T5 (no V2 extension) and Leela's raw policy: as fast as T5 plus one Leela evaluation.
# Auto: LIK_AUTO["nvidia"] when lc0 runs on an NVIDIA GPU, LIK_AUTO["cpu"] otherwise, T5 when Leela is missing or has failed in this process.
LIK_MODELS = ["T5", "Auto", "H8", "H8s", "H6b"]
LIK_NETS = {"H8": "corrector_sf_ft_level2_sfa_pol_weights.json", "H8s": "corrector_sf_ft_level2_sfa_pol_weights.json",
            "H6b": "corrector_hyb_v6b_weights.json"}   # H6b alone (no F1g mix)
LIK_V2 = {"H8", "H6b"}   # the models that read the V2 values
# Auto's choice per lc0 device, set by the calibration's adoption rule (the development log, 12 November): candidate - T5 paired |error| difference < 0 with
# the 95 % upper bound < 0 in Classical. 2026-11-12: none passed (H6b -2.7 [-6.3; +0.7], H8 -1.9 [-5.3; +1.4], H8s -1.2 [-3.7; +1.3]), so
# Auto = T5 everywhere; H8 / H8s / H6b stay selectable by name (for tests and a future calibration).
LIK_AUTO = {"nvidia": "T5", "cpu": "T5"}
CORRECTOR_LIK = {}
try:
    if _NNC is not None:
        for _k, _f in LIK_NETS.items():
            if os.path.exists(HERE / "nn" / _f):
                _same = [c for k, c in CORRECTOR_LIK.items() if LIK_NETS[k] == _f]   # H8s: the same network object as H8
                CORRECTOR_LIK[_k] = _same[0] if _same else _NNC.Corrector(str(HERE / "nn" / _f))
except Exception:  # noqa: BLE001
    CORRECTOR_LIK = {}
opts["lik_model"] = os.environ.get("TOBY_LIK_MODEL", "T5") if os.environ.get("TOBY_LIK_MODEL", "T5") in LIK_MODELS else "T5"


def lik_model():
    """The network 'toby likelihood' uses now: "T5", "H8", "H8s" or "H6b". The last three only in Classical with the fine-tuned Maia, and only while Leela
    (lc0 + its network) is there and has not failed in this process (LEELA_BROKEN is never cleared: once lc0 fails, T5 for the process's life)."""
    m = opts.get("lik_model", "T5")
    if m == "T5" or opts["mode"] != "Classical" or maia_ft is None or not opts.get("maia_ft", True) or CORRECTOR_SF_FT is None:
        return "T5"
    if not leela_found():
        return "T5"
    if m == "Auto":
        m = LIK_AUTO["nvidia" if (HAS_NVIDIA and lc0_backend().startswith(NVIDIA_BACKENDS)) else "cpu"]
    return m if m in CORRECTOR_LIK else "T5"


_LIK_STATIC = []
LIK_SAID = [False]


def likelihood_unavailable():
    """Why 'toby likelihood' cannot be served in this process, or None. Known without loading anything (at 'uci'): Maia-3's code and
    weights (a package without Maia, e.g. the core package), the Stockfish network and Stockfish; after the loading, a Maia that failed."""
    if not TOBY1_SHIPPED:   # 2.2.1: no Maia in this Toby; never looked for elsewhere (Hugging Face cache, download)
        return TOBY1_ABSENT
    if not _LIK_STATIC:
        why = None
        try:
            import importlib.util
            if MAIA_REPO not in sys.path and os.path.isdir(MAIA_REPO):
                sys.path.insert(0, MAIA_REPO)
            if importlib.util.find_spec("maia3") is None:
                why = "Maia-3 is not in this package"
            elif maia_checkpoint() is None:
                why = "Maia-3's weights are not in this package (maia/maia3-79m.pt)"
        except Exception as exc:  # noqa: BLE001
            why = f"Maia-3 cannot be loaded ({type(exc).__name__})"
        if why is None and CORRECTOR_SF is None:
            why = "the Stockfish network is missing (nn/corrector_sf_weights.json)"
        if why is None and not os.path.exists(SF_PATH):
            why = "Stockfish is missing"
        _LIK_STATIC.append(why)
    if _LIK_STATIC[0]:
        return _LIK_STATIC[0]
    if ready.is_set() and maia is None:
        return load_error or "Maia not loaded"
    return None


def likelihood(u, ratings, opp=None, clock="current", dist=None, show_sf=False, bd=None, eng=None, emit=None, spec_ok=True):
    """Probability that a player of each rating plays the move u in the current position (Siby's Bayesian Elo estimate).
    One Stockfish search (it does not depend on the rating), then Maia (with the previous positions) and the Stockfish network for
    each rating; opponent not given, or given (opp: the opponent's rating, as in game analysis with both ratings); clock: (mover's seconds,
    opponent's seconds) before the move, None (unknown) or "current" (CLOCK), used by a Maia time model only (Maia C v2), else ignored. Prints 'info string toby likelihood <u> <r1> <p1> <r2> <p2> ...' then 'info string toby likelihood end model <T5|H8|H8s|H6b>'
    (or 'info string toby likelihood error <reason>' before the end line). Ratings are in the Rating Source, like White / Black Rating.
    The model is lik_model() (option Likelihood Model): with H8 / H6b, Leela's raw policy first (if lc0 fails: T5 now and for the rest of the
    process), then the V2 Stockfish values (150k over every move, 250k on the candidates of each rating), exactly as the hybrid search path;
    with H8s, the network H8 on the 300k search of T5 (no V2) and Leela's raw policy.
    dist: a rating at which the whole distribution is printed too, Toby's ('info string toby dist <r> <uci> <p> ...') and Maia's alone
    ('info string toby maia <r> <uci> <p> ...'), every legal move, most likely first. show_sf: Stockfish's own values from the same search
    (centipawns or mate for the side to move): 'info string toby sf depth <d> nodes <n>', 'info string toby sf best <uci> cp|mate <v> pv ...',
    'info string toby sf played <uci> cp|mate <v> pv ...', 'info string toby sf all <uci> cp|mate <v> ...' (every legal move, best first)."""
    global sf_engine, lc0, LEELA_BROKEN
    b = (bd if bd is not None else board).copy()
    emit = emit or out   # a review batch collects each answer, then prints the answers in order (toby batch)
    model = "T5"
    try:
        why_not = likelihood_unavailable()   # known at once (no wait for the loading): said, then the end line
        if why_not:
            raise RuntimeError(why_not)
        wait_maia()   # loaded at the first request in a Toby 2 process (2.0.2)
        if maia is None:
            raise RuntimeError(load_error or "Maia not loaded")
        if CORRECTOR_SF is None or not os.path.exists(SF_PATH):
            raise RuntimeError("Stockfish or its network missing")
        if eng is None:
            if sf_engine is None:
                sf_engine = StockfishCalc()
            eng = sf_engine
        wait_maia_ft()
        model = lik_model()
        pol = {}
        if model != "T5":   # Leela's raw policy (one evaluation, as the hybrid search path); lc0 failing: T5 from now on
            try:
                pol = leela_raw_policy(b)
            except Exception as lexc:  # noqa: BLE001
                LEELA_BROKEN = True
                stop_all_lc0()
                lc0 = None
                emit(f"info string Leela unavailable ({lexc}): toby likelihood uses T5 from now on")
                model, pol = "T5", {}
        exp = model in LIK_V2   # T5 / H8s: Stockfish 300k over every move (no V2); H8 / H6b: V2 values
        sf = eng.evaluate(b, V2_BASE if exp else SF_TRAIN_NODES)
        if sf is None:
            raise RuntimeError("stopped")
        if u not in sf:
            raise RuntimeError(f"illegal move {u}")
        if show_sf:
            meta = eng.last or {}
            mates, pvs = meta.get("mate", {}), meta.get("pv", {})
            order = sorted(sf, key=lambda m: -sf[m])
            emit(f"info string toby sf depth {meta.get('depth', 0)} nodes {meta.get('nodes', 0)}")
            for tag, m in (("best", order[0]), ("played", u)):
                emit(f"info string toby sf {tag} {m} {_score_txt(sf[m], mates.get(m))} pv {pvs.get(m, m)}")
            emit("info string toby sf all " + " ".join(f"{m} {_score_txt(sf[m], mates.get(m))}" for m in order))
        lee, root = _NNC.sf_pseudo_leela(sf)
        _FT.force = bool(maia_ft is not None and opts["mode"] == "Classical" and opts.get("maia_ft", True) and CORRECTOR_SF_FT is not None)
        cor = CORRECTOR_LIK[model] if model != "T5" else (CORRECTOR_SF_FT if _FT.force else CORRECTOR_SF)
        base_sf = dict(sf)
        use_pm2 = cor.meta.get("use_pm2")
        budget = budget_input(cor, SF_TRAIN_NODES)
        pairs = []   # every Maia input of the loop below, computed in one batch
        for r in list(ratings) + ([dist] if dist is not None else []):
            E = strength_E(r, opts["source"])
            x = rule_params(E)[0]
            xo = rule_params(min(max(strength_E(opp, opts["source"]), E - MAX_ELO_GAP), E + MAX_ELO_GAP))[0] if opp else None
            pairs.append((x, xo))
            if use_pm2:
                pairs.append((maia_own_rating(r), None))
        maia_batch(b, pairs, history=True, clock=clock)
        pre = {}
        sp = spec_engine() if (exp and spec_ok) else None
        if sp is not None and _SPEC["busy"].acquire(blocking=False):   # the grid's distinct extensions, two at a time (same values)
            try:
                sets = []
                for r in list(ratings) + ([dist] if dist is not None else []):
                    E_ = strength_E(r, opts["source"])
                    x_ = rule_params(E_)[0]
                    xo_ = rule_params(min(max(strength_E(opp, opts["source"]), E_ - MAX_ELO_GAP), E_ + MAX_ELO_GAP))[0] if opp else None
                    c_ = v2_candidates(maia_policy(b, x_, xo_, history=True, clock=clock), base_sf)
                    if len(c_) >= 2 and tuple(c_) not in sets:
                        sets.append(tuple(c_))
                qq = queue.Queue()
                for c_ in sets:
                    qq.put(c_)

                def _ext(e_):
                    while not stop_flag.is_set():
                        try:
                            c_ = qq.get_nowait()
                        except queue.Empty:
                            return
                        v_ = e_.extend(b, list(c_), V2_EXT)
                        if v_ is not None:
                            pre[c_] = v_
                ths = [WThread(target=_ext, args=(e_,), daemon=True) for e_ in (eng, sp)]
                [t_.start() for t_ in ths]
                [t_.join() for t_ in ths]
            finally:
                _SPEC["busy"].release()
        res = []
        for r in list(ratings) + ([dist] if dist is not None else []):
            if stop_flag.is_set():
                raise RuntimeError("stopped")
            E = strength_E(r, opts["source"])
            x = rule_params(E)[0]
            Eo = xo = None
            if opp:
                Eo = min(max(strength_E(opp, opts["source"]), E - MAX_ELO_GAP), E + MAX_ELO_GAP)
                xo = rule_params(Eo)[0]
            pm = maia_policy(b, x, xo, history=True, clock=clock)
            pm2 = maia_policy(b, maia_own_rating(r), history=True, clock=clock) if use_pm2 else None
            v2i = {}
            if exp:   # V2: the candidates come from Maia at this rating (one extension per distinct set of candidates, cached)
                sv2 = sf_v2(b, pm, base_sf, eng, pre=pre, info=v2i)
                if sv2 is None:
                    raise RuntimeError("stopped")
                lee, root = _NNC.sf_pseudo_leela(sv2)
            moves, ctx, mv, base = _NNC.build_inputs(b, _r6(pm), lee, pol, {}, root, x, E, opts["mode"].lower(), venue(),
                                                     E_opp=Eo, pm2=_r6(pm2), history=True, sf_budget=budget, v2info=v2i.get("v2info"))
            d = cor.distribution(ctx, mv, base, moves)
            if dist is not None and r == dist and len(res) == len(ratings):   # the extra pass for the distribution
                emit(f"info string toby dist {r} " + " ".join(f"{m} {p:.6g}" for m, p in sorted(d.items(), key=lambda kv: -kv[1])))
                emit(f"info string toby maia {r} " + " ".join(f"{m} {p:.6g}" for m, p in sorted(pm.items(), key=lambda kv: -kv[1])))
                continue
            res.append(f"{r} {d.get(u, 0.0):.6g}")
        emit(f"info string toby likelihood {u} " + " ".join(res))
    except Exception as exc:  # noqa: BLE001
        emit(f"info string toby likelihood error {exc}")
    finally:
        _FT.force = None
    emit(f"info string toby likelihood end model {model}")


# ----------------------------------------------------------------------------------------------------- 'toby level' (Toby 2 level head, no Maia)
# the method page. Per half-move, Toby 2's body B at two fixed anchors (1600, 2400) and the level head give the move's
# log-likelihood curve over the ratings (natural log, summable over a player's moves). Siby sums the curves and applies the calibration line.
_LH_DIR = HERE / "nn" / "level_t2" / "lh"


def _first_existing(env, cands):
    if os.environ.get(env):
        return Path(os.environ[env])
    for c in cands:
        if c.exists():
            return c
    return cands[0]


LEVEL_HEAD = _first_existing("TOBY2_LEVEL_HEAD", [HERE / "nets" / "toby2_level.pt", _LH_DIR / "frozen" / "toby2_level_a2.pt"])
LEVEL_CAL = _first_existing("TOBY2_LEVEL_CAL", [HERE / "nets" / "toby2_level_cal.json", _LH_DIR / "frozen" / "calibration.json"])
LEVEL_MODEL = "toby2-level-v1"
LEVEL_GRID = list(range(1200, 3001, 50))   # the answer's ratings (the head's grid is 1200-3000 step 25; step 50 keeps E_A within ~1 Elo)
LEVEL = {"svc": None, "error": None, "lock": threading.Lock(), "net_lock": threading.Lock(), "mod": None}
TC_CADENCE = {"Classical": "classical", "Rapid": "rapid", "Blitz": "blitz"}


def _level_mod():
    if LEVEL["mod"] is None:
        if str(_LH_DIR) not in sys.path:
            sys.path.insert(0, str(_LH_DIR))
        import levelgen as _lg
        LEVEL["mod"] = _lg
    return LEVEL["mod"]


def level_unavailable():
    """Why 'toby level' cannot be served here, or None (known without loading the network)."""
    if _T2MOD is None:
        return T2.get("error") or "Toby 2 module missing"
    if T2_WEIGHTS is None or not T2_WEIGHTS.exists():
        return "Toby 2 network missing"
    if not LEVEL_HEAD.exists():
        return f"level head missing ({LEVEL_HEAD.name})"
    try:
        _level_mod()
    except Exception as exc:  # noqa: BLE001
        return f"level module: {exc}"
    return None


def level_service():
    with LEVEL["lock"]:
        if LEVEL["svc"] is None and LEVEL["error"] is None:
            try:
                net = _load_t2()
                if net is None:
                    raise RuntimeError(T2.get("error") or "Toby 2 network unavailable")
                LEVEL["svc"] = _level_mod().GenService(net, str(LEVEL_HEAD))
            except Exception as exc:  # noqa: BLE001
                LEVEL["error"] = f"{type(exc).__name__}: {exc}"
        return LEVEL["svc"]


def level_calibration_lines():
    """One 'info string toby level calibration ...' line per cadence group (calibration file of the frozen variant). Fields: grid, E_A
    (tau, prior mean / sd, correction and its coefficients on the linear-spline basis [1, x, (x-1800)+, (x-2200)+, (x-2500)+] with x
    bounded to [1400, 2800]), E_R (tau, s, drift coefficients on the same basis of the given rating), the result constants c, the per-game
    sd scale, and 'use': which estimator Siby should take from this head for that group (level) or from Toby 1.0 (toby1)."""
    try:
        cal = json.loads(Path(LEVEL_CAL).read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"info string toby level calibration error {exc}"]
    res = []
    for g, c in cal["groups"].items():
        a, r = c["E_A"], c["E_R"]
        use = c.get("use", {"E_A": "level", "E_R": "level"})
        f = lambda xs: " ".join(f"{float(x):.10g}" for x in xs)
        res.append(f"info string toby level calibration model {cal.get('model', LEVEL_MODEL)} group {g} grid 1200 3000 25 "
                   f"ea_tau {a['tau']} ea_mean {a['prior_mean']:.2f} ea_sd {a['prior_sd']:.2f} ea_corr {a['correction']} ea_coef {f(a['coef'])} "
                   f"knots 1800 2200 2500 bounds 1400 2800 er_tau {r['tau']} er_s {r['s']} er_drift {f(r['drift_coef'])} "
                   f"c_er {c['result_c']['E_R']} c_ea {c['result_c']['E_A']} sd_scale {c.get('sd_scale', 1.0):.4f} "
                   f"use_ea {use.get('E_A', 'level')} use_er {use.get('E_R', 'level')}")
    return res


def level_display_lines():
    """One 'info string toby level display model M group g ea_stretch k ea_coef_review c0 .. c4 ea_stretch_review k ea_cap_high e
    ea_cap_low e' line per group whose calibration has 'ea_display' (nn/level_t2/lh/frozen_open_c/ea_display_fields.json, merged into
    nets/toby2_level_cal.json by nn/level_t2/lh/merge_ea_display.py): the E_A display used when no rating is known (E_R unchanged).
    A separate line, printed after the calibration lines: Siby's parsers append unknown words to the previous field, so the calibration
    lines stay byte-identical to 2.2.1's. No 'ea_display' in the file (2.2.1's calibration): no line."""
    try:
        cal = json.loads(Path(LEVEL_CAL).read_text())
    except Exception:  # noqa: BLE001
        return []   # the calibration lines already say the error
    g = lambda x: f"{float(x):.10g}"
    res = []
    for grp, c in cal["groups"].items():
        d = c.get("ea_display")
        if d:
            res.append(f"info string toby level display model {cal.get('model', LEVEL_MODEL)} group {grp} ea_stretch {g(d['ea_stretch'])} "
                       f"ea_coef_review {' '.join(g(x) for x in d['ea_coef_review'])} ea_stretch_review {g(d['ea_stretch_review'])} "
                       f"ea_cap_high {g(d['ea_cap_high'])} ea_cap_low {g(d['ea_cap_low'])}")
    return res


def level_move(u, best=None, played=None, cad=None, dist=None, bd=None, eng=None, emit=None):
    """'toby level move <uci> [best <cp> played <cp>] [cadence c] [dist <elo>]' on the current position (before the move). Prints
    'info string toby level <uci> <r1> <l1> <r2> <l2> ...' (natural log-likelihood of the move at each rating of LEVEL_GRID), with dist
    'info string toby dist <elo> <uci> <p> ...' (the head's move distribution at that rating, values-free, B's two anchor passes), then
    'info string toby level end model toby2-level-v1 values given|searched|none [best <cp> played <cp>]' ('searched': Toby's own
    150k MultiPV search over every move, Toby 2's pass 1; 'none': no Stockfish available, the expected-score term is left out).
    On an error: 'info string toby level error <reason>' before the end line (Siby skips that move)."""
    emit = emit or out
    b = bd if bd is not None else board
    mode = "given"
    try:
        svc = level_service()
        if svc is None:
            raise RuntimeError(LEVEL["error"] or "level head unavailable")
        if cad is None:
            cad = TC_CADENCE.get(opts["mode"], "classical")
        if cad not in TC_CADENCE.values():
            raise ValueError(f"cadence {cad}")
        if chess.Move.from_uci(u) not in b.legal_moves:
            raise ValueError(f"illegal move {u}")
        if best is None or played is None:
            best = played = None
            mode = "none"
            e = eng
            if e is None and os.path.exists(SF_PATH):
                global sf_engine
                if sf_engine is None:
                    sf_engine = StockfishCalc()
                e = sf_engine
            if e is not None:
                v = e.evaluate(b, V2_BASE)
                if v is None:
                    raise RuntimeError("stopped")
                best, played, mode = max(v.values()), v[u], "searched"
        with LEVEL["net_lock"]:
            r = svc.move(b, u, cad, best, played, dist)
        c = r["curve"]
        emit(f"info string toby level {u} " + " ".join(f"{e} {float(c[(e - 1200) // 25]):.4f}" for e in LEVEL_GRID))
        if dist is not None:
            emit(f"info string toby dist {dist} " + " ".join(f"{m} {p:.6g}" for m, p in sorted(r["dist"].items(), key=lambda kv: -kv[1])))
        tail = f" best {int(best)} played {int(played)}" if mode == "searched" else ""
        emit(f"info string toby level end model {LEVEL_MODEL} values {mode}{tail}")
    except Exception as exc:  # noqa: BLE001
        emit(f"info string toby level error {exc}")
        emit(f"info string toby level end model {LEVEL_MODEL} values none")


def parse_level(t):
    """tokens of 'toby level move <uci> ...' -> dict of level_move's arguments (ValueError on a malformed command)."""
    if len(t) < 4 or t[2] != "move":
        raise ValueError("usage: toby level move <uci> [best <cp> played <cp>] [cadence classical|rapid|blitz] [dist <elo>]")
    a = {"u": t[3], "best": None, "played": None, "cad": None, "dist": None}
    rest = t[4:]
    while rest:
        k = rest[0]
        if k in ("best", "played"):
            a[k] = int(float(rest[1]))
        elif k == "cadence":
            a["cad"] = rest[1].lower()
        elif k == "dist":
            a["dist"] = int(float(rest[1]))
        else:
            raise ValueError(f"unknown keyword {k}")
        rest = rest[2:]
    return a


# ----------------------------------------------------------------------------------------------------- review batches (parallel 'toby likelihood')
REVIEW_POOL = []                 # single-thread Stockfish engines of the review batches, kept between batches
_REVIEW_LOCK = threading.Lock()


def review_workers():
    """Option Review Workers, else the number of cores - 1 (at least 1, at most 16): one single-thread Stockfish each."""
    if opts.get("review_workers"):
        return opts["review_workers"]
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        n = os.cpu_count() or 2
    return max(1, min(16, n - 1))


def run_batch(items):
    """'toby batch begin' ... 'toby batch end': the queued 'toby likelihood' commands run in parallel, one single-thread Stockfish per
    worker; each answer is the block a single 'toby likelihood' prints, and the blocks are printed in the order of the commands (each as
    soon as it and the ones before are done), then 'info string toby batch end'. The workers share Maia (one call at a time).
    'toby level move' commands may be queued too (kind "level"): Maia is not needed for them, they share Toby 2's network (one call at
    a time) and use the worker's Stockfish only when no values are given."""
    try:
        if any(it.get("kind") != "level" for it in items):
            wait_maia(ft=True)
        k = max(1, min(review_workers(), len(items)))
        with _REVIEW_LOCK:
            while len(REVIEW_POOL) < k and os.path.exists(SF_PATH):
                REVIEW_POOL.append(StockfishCalc(fixed_threads=1))
        free = queue.Queue()
        for e in REVIEW_POOL[:k]:
            free.put(e)
        for _ in range(k - len(REVIEW_POOL[:k])):   # no Stockfish: level items with values still run (a search-less worker)
            free.put(None)
        done = [None] * len(items)
        nxt = [0]
        plock = threading.Lock()

        def flush():
            with plock:
                while nxt[0] < len(items) and done[nxt[0]] is not None:
                    for line in done[nxt[0]]:
                        out(line)
                    nxt[0] += 1

        def one(i):
            it = items[i]
            buf = []
            e = free.get()
            try:
                if it.get("kind") == "level":
                    level_move(it["u"], it["best"], it["played"], it["cad"], it["dist"], bd=it["board"], eng=e, emit=buf.append)
                else:
                    likelihood(it["u"], it["ratings"], it["opp"], it["clock"], it["dist"], it["sf"], bd=it["board"], eng=e, emit=buf.append, spec_ok=False)
            finally:
                free.put(e)
            done[i] = buf
            flush()

        ths = []
        q = queue.Queue()
        for i in range(len(items)):
            q.put(i)

        def worker_loop():
            while not stop_flag.is_set():
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    return
                one(i)

        for _ in range(k):
            th = WThread(target=worker_loop, daemon=True)
            th.start()
            ths.append(th)
        for th in ths:
            th.join()
        for i in range(len(items)):   # stopped: the answers not computed say so, in order
            if done[i] is None:
                done[i] = ([f"info string toby level error stopped", f"info string toby level end model {LEVEL_MODEL} values none"]
                           if items[i].get("kind") == "level" else
                           [f"info string toby likelihood error stopped", f"info string toby likelihood end model {lik_model()}"])
        flush()
    except Exception as exc:  # noqa: BLE001
        out(f"info string toby batch error {exc}")
    out("info string toby batch end")


def stop_review_pool():
    while REVIEW_POOL:
        REVIEW_POOL.pop().quit()
    if _SPEC["eng"] is not None:
        _SPEC["eng"].quit()
        _SPEC["eng"] = None
    for k in ("eng", "spec3"):   # Toby 2's own Stockfish engines
        e = T2.get(k) if isinstance(globals().get("T2"), dict) else None
        if e is not None:
            try:
                e.quit()
            except Exception:  # noqa: BLE001
                pass
            T2[k] = None


atexit.register(stop_review_pool)


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
    if TOBY1_SHIPPED:   # Endgame Evaluation acts inside Toby 1.0's evaluation module only
        out("option name Endgame Evaluation type combo default M var Off " + " ".join(f"var {c}" for c in ENDGAME_CANDIDATES))
    out("option name Provisional Lines type check default true")
    out("option name Leela Preload type check default false")
    out("option name Deepening type check default true")   # go infinite: after the final block, the 3 most probable moves and the best move view get the next position's own evaluation (blended), one updated block   # true: lc0 starts right after readyok where Leela is used (analysis Toby)
    if CORRECTOR_SF is not None and not TOBY1_SHIPPED:   # 2.2.1: Toby 1.0's options are not advertised (still accepted, see toby1_value)
        out(f"option name Stockfish Threads type spin default {opts['sf_threads']} min 0 max 256")
    elif CORRECTOR_SF is not None:
        cur = "Fast" if (opts.get("hybrid") or opts["calculator"] == "Stockfish") else "Accurate"
        out(f"option name Precision type combo default {cur} " + " ".join(f"var {p}" for p in PRECISIONS))
        # Precision = Accurate (Calculator Auto: Leela in Classical with an NVIDIA GPU, Stockfish elsewhere) / Fast (Stockfish always).
        # The older option Calculator (Auto / Leela / Stockfish) is still accepted, not advertised; Calculator = Leela forces Leela.
        out(f"option name Stockfish Threads type spin default {opts['sf_threads']} min 0 max 256")
        out(f"option name Review Workers type spin default {opts.get('review_workers', 0)} min 0 max 64")
        out(f"option name Stockfish Nodes type spin default {opts['sf_nodes']} min 0 max 1000000000")
        out(f"option name Maia Threads type spin default {opts['maia_threads']} min 0 max 256")
    if CORRECTOR_LIK and TOBY1_SHIPPED:   # network of 'toby likelihood' in Classical (Siby's level estimate): T5 (default), Auto (LIK_AUTO per lc0 device), H8, H8s, H6b
        out(f"option name Likelihood Model type combo default {opts['lik_model']} " + " ".join(f"var {m}" for m in LIK_MODELS))
    if TOBY1_SHIPPED and os.path.exists(MAIA_FT_PATH) and (CORRECTOR_SF_FT is not None or CORRECTOR_HSF_FT is not None):
        out(f"option name Fine-tuned Maia type check default {'true' if opts['maia_ft'] else 'false'}")
    if CORRECTOR is not None and TOBY1_SHIPPED:
        out(f"option name Move Model type combo default {opts['move_model']} " + " ".join(f"var {m}" for m in MOVE_MODELS))
    if _T2MOD is not None and T2_WEIGHTS is not None and T2_WEIGHTS.exists():   # Toby 2 (network B)
        if TOBY1_SHIPPED:
            out(f"option name Move Network type combo default {opts['move_network']} " + " ".join(f"var {m}" for m in MOVE_NETWORKS))
        offered = [m for m in T2_PRODUCTS if m != "Accurate" or t2_accurate_offered() or opts["t2_product"] == "Accurate"]
        out(f"option name Toby 2 Product type combo default {opts['t2_product']} " + " ".join(f"var {m}" for m in offered))
        out(f"option name Leela Head type combo default {opts['t2_head']} " + " ".join(f"var {m}" for m in T2_HEADS))
        cands = [m for m in _T2MOD.CANDIDATE_SOURCES if TOBY1_SHIPPED or not toby1_value("Candidates", m)]
        cdef = opts["t2_candidates"] if opts["t2_candidates"] in cands else cands[0]
        out(f"option name Candidates type combo default {cdef} " + " ".join(f"var {m}" for m in cands))
        out(f"option name Network Device type combo default {opts['t2_device']} " + " ".join(f"var {m}" for m in T2_DEVICES))
        out(f"option name Network Threads type spin default {opts['t2_threads']} min 0 max 64")
        mev = [m for m in MOVE_EVALS if TOBY1_SHIPPED or not toby1_value("Move Evaluations", m)]
        pev = [m for m in POSITION_EVALS if TOBY1_SHIPPED or not toby1_value("Position Evaluation", m)]
        out(f"option name Move Evaluations type combo default {opts['t2_move_evals']} " + " ".join(f"var {m}" for m in mev))
        out(f"option name Position Evaluation type combo default {opts['t2_position_eval']} " + " ".join(f"var {m}" for m in pev))
        out(f"option name Human Blunder Check type check default {'true' if opts['t2_blunder_check'] else 'false'}")
        out(f"option name Base Time type spin default {opts['t2_base']} min 0 max 36000")
        out(f"option name Increment type spin default {opts['t2_inc']} min -1 max 600")
    # the "Mode" option (Prediction | Human) is deliberately not advertised: it only matters for a bot build


def toby1_value(nm, val):
    """True for an option value that selects Toby 1.0 or Maia (Move Network = Toby 1.0, Candidates = Maia C, Move Evaluations = Toby 1.0,
    Position Evaluation = Toby 1.0 / Mix). Without Toby 1.0 (TOBY1_SHIPPED false) such a value is ignored: the current one stays."""
    v = val.lower()
    return ((nm == "Move Network" and v == "toby 1.0") or (nm == "Candidates" and v == "maia c")
            or (nm == "Move Evaluations" and v == "toby 1.0") or (nm == "Position Evaluation" and v in ("toby 1.0", "mix")))


def set_option(t):
    global lc0
    i = t.index("name")
    j = t.index("value") if "value" in t else len(t)
    nm, val = " ".join(t[i + 1:j]), " ".join(t[j + 1:])
    nm = OPTION_ALIASES.get(nm, nm)
    val = VALUE_ALIASES.get(val, val)
    if not TOBY1_SHIPPED and toby1_value(nm, val):   # 2.2.1: a value that would need Toby 1.0 / Maia is accepted and ignored (GUIs keep working)
        return
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
    elif nm == "Provisional Lines":
        opts["provisional"] = val.lower() == "true"
    elif nm == "Deepening":
        opts["deepening"] = val.lower() == "true"
    elif nm == "Likelihood Model" and val in LIK_MODELS:
        opts["lik_model"] = val
    elif nm == "Leela Preload":   # false (default): lc0 starts at the first search that needs it; never in a likelihood-only process
        opts["leela_preload"] = val.lower() == "true"
    elif nm == "Endgame Evaluation" and val in ["Off"] + ENDGAME_CANDIDATES:
        opts["wdl_endgame"] = val
    elif nm == "UCI_ShowWDL":
        opts["UCI_ShowWDL"] = val.lower() == "true"
    elif nm == "Rating Source" and val in SOURCES:
        opts["source"] = val
    elif nm == "Time Control" and val in TIME_CONTROLS:
        opts["mode"] = val   # lc0 starts at the first search that needs it (start_leela_for_search)
    elif nm == "Mode":
        opts["move_choice"] = "human" if val.lower().startswith("human") else "prediction"
    elif nm == "Move Model" and val in MOVE_MODELS:
        opts["move_model"] = val
    elif nm == "Maia Threads":
        opts["maia_threads"] = max(0, int(float(val)))
        if opts["maia_threads"] == 0 and os.environ.get("TOBY_TORCH_THREADS", "0") not in ("", "0"):   # a GUI resending the default keeps the environment's value
            opts["maia_threads"] = int(os.environ["TOBY_TORCH_THREADS"])
        set_maia_threads(opts["maia_threads"])
    elif nm == "Precision" and val in PRECISIONS + ["Hybrid"]:
        if val in ("Fast", "Hybrid"):   # Stockfish calculator (V2 values) + one Leela evaluation per move (Classical); lc0 starts at the first search
            opts["hybrid"] = True
            opts["calculator"] = "Stockfish"
            if sf_engine is None:
                threading.Thread(target=preload_lc0, daemon=True).start()   # starts and warms up Stockfish
        else:
            opts["hybrid"] = False
            set_option(["name", "Calculator", "value", "Auto"])
    elif nm == "Calculator" and val in CALCULATORS:
        opts["calculator"] = val
        opts["hybrid"] = False
        if val == "Stockfish":
            stop_all_lc0()   # no Leela at all with the Stockfish calculator
            lc0 = None
            if sf_engine is None:
                threading.Thread(target=preload_lc0, daemon=True).start()   # starts and warms up Stockfish
        else:
            stop_stockfish()   # lc0 starts at the first search that needs it
    elif nm == "Fine-tuned Maia":
        opts["maia_ft"] = val.lower() == "true"
    elif nm == "Review Workers":
        opts["review_workers"] = max(0, min(64, int(float(val))))
    elif nm == "Stockfish Threads":
        opts["sf_threads"] = max(0, int(float(val)))
    elif nm == "Stockfish Nodes":
        opts["sf_nodes"] = max(0, int(float(val)))
    elif nm == "Display" and val in DISPLAYS:
        opts["display"] = val
    elif nm == "Move Network" and val in MOVE_NETWORKS:
        if val != opts["move_network"]:   # a GUI resending the default is not a choice
            opts["move_network_set"] = True
        opts["move_network"] = val
        if maia_wanted():
            ensure_maia()
    elif nm == "Candidates" and _T2MOD is not None and val in _T2MOD.CANDIDATE_SOURCES:
        if val != opts["t2_candidates"]:   # a GUI resending the default does not override the product file
            opts["t2_candidates_set"] = True
        opts["t2_candidates"] = val
        if maia_wanted():
            ensure_maia()
    elif nm == "Toby 2 Product" and {"Fast": "Standard"}.get(val, val) in T2_PRODUCTS:
        val = {"Fast": "Standard"}.get(val, val)   # the protocol's name of the Standard product
        if val != opts["t2_product"] and not os.environ.get("TOBY2_CONFIG"):   # reloaded with the other product file at the next search
            opts["t2_product"] = val
            T2.update(net=None, search=None, cfg=None, error=None)
    elif nm == "Leela Head" and val in T2_HEADS:
        opts["t2_head"] = val
    elif nm == "Network Device" and val in T2_DEVICES:
        if val != opts["t2_device"] and T2["net"] is not None:   # reloaded at the next search on the new device
            T2["net"] = T2["search"] = None
        opts["t2_device"] = val
    elif nm == "Network Threads":
        opts["t2_threads"] = max(0, int(float(val)))
        if T2["net"] is not None and T2["net"].device == "cpu" and opts["t2_threads"]:
            import torch
            torch.set_num_threads(opts["t2_threads"])
    elif nm == "Move Evaluations":   # Toby 2 / Toby 1.0 / Off; the 2.0.1 check values still work: true = Toby 2, false = Off
        v_ = {"true": "Toby 2", "false": "Off"}.get(val.lower(), val)
        if v_ in MOVE_EVALS:
            opts["t2_move_evals"] = v_
            if maia_wanted():
                ensure_maia()
    elif nm == "Position Evaluation" and val in POSITION_EVALS:
        opts["t2_position_eval"] = val
        if maia_wanted():
            ensure_maia()
    elif nm == "Human Blunder Check":
        opts["t2_blunder_check"] = val.lower() == "true"
    elif nm == "Base Time":
        opts["t2_base"] = max(0, int(float(val)))
    elif nm == "Increment":
        opts["t2_inc"] = max(-1, int(float(val)))


def main():
    global worker, CLOCK
    batch = None   # 'toby batch begin': the 'toby likelihood' commands are queued (with their position) until 'toby batch end'
    if maia_wanted():   # Toby 1.0 (or TOBY_MAIA_EAGER=1): Maia loads at once, as before; Toby 2 loads it only when something uses it
        ensure_maia()
    if t2_on():
        threading.Thread(target=load_t2, daemon=True).start()   # network B in the background (the first search waits for it if needed)
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=preload_lc0, kwargs={"leela": False}, daemon=True).start()   # Stockfish only: lc0 waits for the GUI's options
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
            out("id author Yosha Iglesias")
            out(f"info string toby version {VERSION}")
            uci_options()
            lik_no = likelihood_unavailable()
            if lik_no:   # Siby skips its level estimate at once (it checks the capabilities below) instead of waiting for answers
                out("info string toby capabilities")   # a capabilities line with no likelihood word: Siby turns its level estimate
                out(f"info string toby likelihood unavailable ({lik_no})")   # off at once (no line at all = an older Toby: it waits)
                LIK_SAID[0] = True   # (batch and maia-device need Maia too: not listed)
            else:
                out("info string toby capabilities likelihood")   # GUIs (Siby) check this before sending 'toby likelihood' (an older Toby ignores it silently)
                out("info string toby capabilities likelihood-opp")
                out("info string toby capabilities likelihood-clk")
                out("info string toby capabilities likelihood-sf")     # 'sf': Stockfish's own values of the same search (best, played, every move)
                out("info string toby capabilities likelihood-dist")
                out("info string toby capabilities batch")
                out("info string toby capabilities maia-device")   # 'info string toby maia device cuda|cpu' before each readyok   # 'toby batch begin' / 'toby batch end': the 'toby likelihood' commands between run in parallel   # 'dist <r>': Toby's and Maia's whole distributions at one rating   # 'toby likelihood <uci> clk <mover s> <opponent s> ...': clocks (used by a time model)   # 'toby likelihood <uci> opp <rating> ...': the opponent's rating given
            missing = [n for n in RELEASE_FILES if not (HERE / "nn" / n).exists()]
            if missing and os.path.exists(MAIA_FT_PATH):   # a package without the networks of the sealed test v5: say it, never fall back silently
                out("info string Toby: WARNING files missing in nn/: " + ", ".join(missing) + " (Classical uses earlier networks; reinstall Toby)")
            lvl_no = level_unavailable()
            if lvl_no is None:   # 'toby level move ...' (Toby 2 level head, no Maia); Siby falls back to 'toby likelihood' without it
                out("info string toby capabilities level")
            else:
                out(f"info string toby level unavailable ({lvl_no})")
            out("uciok")
            if lvl_no is None:   # the frozen calibration of the level head, once, keyed by the model name (Siby applies it)
                for line in level_calibration_lines() + level_display_lines():   # display lines after the calibration lines
                    out(line)
        elif c == "isready":   # lc0 is never started here: a process that only answers 'toby likelihood' / 'dist' (Siby's review) never needs it
            try:   # Toby 2: network B loaded, the Accurate probe settled, Stockfish started (the first search then answers at once)
                t2_prepare()
            except Exception:  # noqa: BLE001   (any error is kept in T2 and said at the search, as before)
                pass
            if maia_wanted() or _MAIA_START["started"]:   # Maia only when the options use it, or already loading (2.0.2)
                wait_maia()
                if not LIK_SAID[0] and likelihood_unavailable():   # Maia advertised at 'uci' but failed to load: said once
                    LIK_SAID[0] = True
                    out(f"info string toby likelihood unavailable ({likelihood_unavailable()})")
                if opts["mode"] == "Classical" and opts.get("maia_ft", True) and os.path.exists(MAIA_FT_PATH):
                    ft_loaded.wait(180)   # the fine-tuned Maia too (a copy of the loaded one: about a second), so that the first search is immediate
            t2_say_missing()
            if maia is not None:   # where Maia runs (GUIs give it 1 thread on a GPU)
                out(f"info string toby maia device {'cuda' if str(maia[0].cfg.device).startswith('cuda') else 'cpu'}")
            elif not _MAIA_START["started"] and likelihood_unavailable() is None:   # not loaded yet: where it will run when loaded
                out(f"info string toby maia device {'cuda' if maia_device().startswith('cuda') else 'cpu'}")
            match_priority()   # the GUI may have changed Toby's priority since its engines started
            if not STARTUP.get("said"):
                STARTUP["said"] = True
                t0 = STARTUP["t0"]
                parts = [f"{k} {STARTUP[k] - t0:.2f}s" for k in ("torch", "maia", "ft", "toby2") if k in STARTUP]
                out(f"info string toby timing since launch: {time.time() - t0:.2f}s at the first readyok ({', '.join(parts)})")
            out("readyok")
            if opts.get("leela_preload"):   # option Leela Preload (an analysis Toby): lc0 loads now, in the background, where Leela is used
                start_leela_for_search()
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
            CLOCK = None
        elif c == "go":
            PREFETCH_ABORT.set()
            if worker is not None and worker.is_alive():   # a new search never starts while the previous one still runs (it stops within about 50 ms)
                stop_flag.set()
                worker.join(timeout=3)
            stop_flag.clear()
            mt = time_budget(t, board.turn)
            CLOCK = go_clock(t, board.turn)
            try:
                _ik = "winc" if board.turn == chess.WHITE else "binc"
                T2_GO_INC[0] = float(t[t.index(_ik) + 1]) / 1000 if _ik in t else None
            except (ValueError, IndexError):
                T2_GO_INC[0] = None
            start_leela_for_search()   # lc0 starts lazily, at the first search that needs it, in the background while Maia runs
            HEART["t0"] = HEART["last"] = time.time()
            worker = WThread(target=search_guarded, args=(mt, "infinite" in t), daemon=True)
            worker.start()
        elif c == "toby" and len(t) >= 3 and t[1] == "batch" and t[2] == "begin":
            batch = []
        elif c == "toby" and len(t) >= 3 and t[1] == "batch" and t[2] == "end":
            items, batch = batch or [], None
            PREFETCH_ABORT.set()
            if worker is not None and worker.is_alive():
                stop_flag.set()
                worker.join(timeout=3)
            stop_flag.clear()
            worker = WThread(target=run_batch, args=(items,), daemon=True)
            worker.start()
        elif c == "toby" and len(t) >= 2 and t[1] == "level" and batch is not None:   # queued: same arguments as below
            try:
                la = parse_level(t)
                batch.append({"kind": "level", **la, "board": board.copy()})
            except (ValueError, IndexError) as exc:
                out(f"info string toby level error {exc}")
                out(f"info string toby level end model {LEVEL_MODEL} values none")
        elif c == "toby" and len(t) >= 2 and t[1] == "level":   # toby level move <uci> [best <cp> played <cp>] [cadence c] [dist <elo>]
            PREFETCH_ABORT.set()
            if worker is not None and worker.is_alive():
                stop_flag.set()
                worker.join(timeout=3)
            stop_flag.clear()
            try:
                la = parse_level(t)
            except (ValueError, IndexError) as exc:
                out(f"info string toby level error {exc}")
                out(f"info string toby level end model {LEVEL_MODEL} values none")
            else:
                worker = WThread(target=level_move, args=(la["u"], la["best"], la["played"], la["cad"], la["dist"]),
                                          kwargs={"bd": board.copy()}, daemon=True)
                worker.start()
        elif c == "toby" and len(t) >= 4 and t[1] == "likelihood" and batch is not None:   # queued: same arguments as below
            opp_r, clk_r, dist_r, show_sf, rest = None, None, None, False, t[3:]
            try:
                while rest and rest[0] in ("clk", "opp", "sf", "dist"):
                    if rest[0] == "clk":
                        clk_r, rest = (float(rest[1]), float(rest[2])), rest[3:]
                    elif rest[0] == "opp":
                        opp_r, rest = int(rest[1]), rest[2:]
                    elif rest[0] == "dist":
                        dist_r, rest = int(rest[1]), rest[2:]
                    else:
                        show_sf, rest = True, rest[1:]
                rs = [int(x) for x in rest]
            except (ValueError, IndexError):
                rs = []
            batch.append({"u": t[2], "ratings": rs, "opp": opp_r, "clock": clk_r, "dist": dist_r, "sf": show_sf, "board": board.copy()})
        elif c == "toby" and len(t) >= 4 and t[1] == "likelihood":   # toby likelihood <uci> [clk <s> <s>] [opp <r>] [sf] [dist <r>] <rating> ... (Siby)
            PREFETCH_ABORT.set()
            if worker is not None and worker.is_alive():
                stop_flag.set()
                worker.join(timeout=3)
            stop_flag.clear()
            opp_r, clk_r, dist_r, show_sf, rest = None, None, None, False, t[3:]
            try:   # keywords in any order before the ratings
                while rest and rest[0] in ("clk", "opp", "sf", "dist"):
                    if rest[0] == "clk":   # seconds on the mover's clock and the opponent's, before the move
                        clk_r, rest = (float(rest[1]), float(rest[2])), rest[3:]
                    elif rest[0] == "opp":
                        opp_r, rest = int(rest[1]), rest[2:]
                    elif rest[0] == "dist":   # the whole distribution at this rating
                        dist_r, rest = int(rest[1]), rest[2:]
                    else:
                        show_sf, rest = True, rest[1:]
                rs = [int(x) for x in rest]
            except (ValueError, IndexError):
                rs = []
            worker = WThread(target=likelihood, args=(t[2], rs, opp_r, clk_r, dist_r, show_sf), daemon=True)
            worker.start()
        elif c == "stop":
            stop_flag.set()
        elif c == "quit":
            stop_flag.set()
            break
    # quit, or stdin closed by the interface: stop the search and every lc0 (the atexit hook and, on Windows, the job object cover the other cases)
    stop_flag.set()
    stop_all_lc0()
    stop_stockfish()
    stop_review_pool()


if __name__ == "__main__":
    main()
