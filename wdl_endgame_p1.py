# -*- coding: utf-8 -*-
"""Endgame correction P1 of Toby's win / draw / loss (the method page, section 11): positions with at most 12 men and no
castling rights, single-rating use; Stockfish path (Fast): Classical, Rapid and Blitz over the board (Rapid and Blitz online: none);
Leela path (Accurate, Leela constants of toby_uci.WDL_SINGLE): Classical only.
Pure Python (no numpy), so that the engine and the checker share exactly the same arithmetic. Inputs only from what the engine already
computes: Stockfish's value (pseudo-Leela Q), Maia's value and Maia's policy from the same value pass (released Maia-3 at the engine's
evaluation rating toby_uci.eval_elo on both sides, position alone): no extra search, no extra network pass, no online service.

The engine's Davidson formula is kept as an offset: with (W, D, L) of toby_uci.wdl_model for the side to move of the evaluated position,
t = ln W - ln L and ln nu = ln D - (ln W + ln L) / 2; then t += sum(a_k f_k), ln nu += sum(b_k g_k), and (W, D, L) is rebuilt.
Features (nn/wdl_ext.design, family cheap+dec), with s0 = exp(-(cp / 100)^2) (cp from Q), h = 1 / (1 + exp((men - m0) / tau)),
s = s0 h, dec = (1 - s0) h, z = (E - 2000) / 1000, mt = clip(material, -9, 9) / 3, eL = logit((1 + Q) / 2):
  odd   mat*s, T_Maia*s, tempo*s, eL*dec, cls[...]*s (material classes, +1 / -1 for the side to move stronger / weaker)
  even  s, |mat|*s, |mat|*s*z, s*z, maia_top*s, maia_entropy*s, dec, dec*z, |eL|*dec
Parameters: nn/wdl_ext_params_v4.json (frozen), per candidate "groups" (Stockfish path) and "accurate" (Leela path), each also
"_opp" for the opponent-rating use (both ratings given), key "<Mode>|<venue>", then "root" (the evaluation of the position) or "after" (the
evaluation of a move, i.e. of the position after it, for the side then to move)."""
import json
import math
from pathlib import Path

VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
ORDER = "kqrbnp"
MAX_MEN = 12
_PARAMS = {}


def load_params(path=None, default=False):
    """The parameter file (default: the one set by load_params(path, default=True), else nn/wdl_ext_params_v4.json next to this module)."""
    global _DEFAULT
    if path is None:
        path = _DEFAULT or str(Path(__file__).resolve().parent / "wdl_ext_params_v4.json")
    key = str(path)
    if key not in _PARAMS:
        _PARAMS[key] = json.loads(Path(path).read_text(encoding="utf-8"))
    if default:
        _DEFAULT = key
    return _PARAMS[key]


_DEFAULT = None


def group_params(cand, mode, venue, path="sf", params=None, opp=False):
    """Parameters of a candidate for a time control and venue ({'root': ..., 'after': ...}): path 'sf' = the Stockfish path (Fast),
    'leela' = the Leela path (Accurate, Classical only); opp = the opponent-rating use (both ratings given; keys with '_opp');
    None where the candidate does not apply."""
    c = (params or load_params())["candidates"].get(cand)
    if not c:
        return None
    key = ("groups" if path == "sf" else "accurate") + ("_opp" if opp else "")
    return c.get(key, {}).get(f"{mode}|{venue}")


def base_consts(cand, mode, venue, path="sf", params=None, opp=False):
    """The candidate's refitted base constants (amendment 8) for a path ('sf' / 'leela'), use (opp: both ratings given), time control and
    venue: a tuple (wL, wM, a2, b2, d2, wU, kE, wD[, z, z^2 draw-level terms]) replacing the engine's, or None (the engine's constants)."""
    c = (params or load_params())["candidates"].get(cand)
    if not c:
        return None
    v = c.get("base", {}).get(("sf" if path == "sf" else "leela") + ("_opp" if opp else ""), {}).get(f"{mode}|{venue}")
    return tuple(v) if v else None


def board_info(fen):
    """(men, material of the side to move minus the opponent's in pawns, signature with the side to move first, castling rights?)"""
    parts = fen.split()
    w, b = [], []
    for ch in parts[0]:
        if ch.isalpha():
            (w if ch.isupper() else b).append(ch.lower())
    stm_white = parts[1] == "w"
    mat = sum(VAL[c] for c in w) - sum(VAL[c] for c in b)
    if not stm_white:
        mat = -mat
    sw = "".join(sorted(w, key=ORDER.index)).upper()
    sb = "".join(sorted(b, key=ORDER.index)).upper()
    sig = f"{sw}-{sb}" if stm_white else f"{sb}-{sw}"
    castling = len(parts) > 2 and parts[2] != "-"
    return len(w) + len(b), mat, sig, castling


def applies(fen):
    men, _, _, castling = board_info(fen)
    return men <= MAX_MEN and not castling


def q_to_cp(q):
    return 111.714640912 * math.tan(1.5620688421 * max(-0.999, min(0.999, q)))


def _lg(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def policy_stats(probs):
    """Top probability and entropy of Maia's policy over the legal moves (list of probabilities), rounded to 5 decimals as the study's data."""
    p = [v for v in probs if v > 0]
    return round(max(p), 5), round(-sum(v * math.log(v) for v in p), 5)


def features(q, maia_ldw, maia_top, maia_ent, fen, E, p):
    men, mat, sig, _ = board_info(fen)
    cp = q_to_cp(q)
    s0 = math.exp(-(cp / 100.0) ** 2)
    h = 1.0 / (1.0 + math.exp((men - p["m0"]) / p["tau"]))
    s, dec = s0 * h, (1 - s0) * h
    z = (E - 2000) / 1000
    mt = max(-9.0, min(9.0, float(mat))) / 3
    l, d, w = [max(1e-4, min(1.0, v)) for v in maia_ldw]
    TM = math.log(w) - math.log(l)
    eL = _lg((1 + q) / 2)
    f = {"mat*s": mt * s, "T_Maia*s": TM * s, "tempo*s": s, "eL*dec": eL * dec,
         "s": s, "|mat|*s": abs(mt) * s, "|mat|*s*z": abs(mt) * s * z, "s*z": s * z, "maia_top*s": maia_top * s,
         "maia_entropy*s": maia_ent * s, "dec": dec, "dec*z": dec * z, "|eL|*dec": abs(eL) * dec}
    for c in p["classes"]:
        st, wk = c.split("-")
        f[f"cls[{c}]*s"] = (1.0 if sig == f"{st}-{wk}" else (-1.0 if sig == f"{wk}-{st}" else 0.0)) * s
    return f


def correct(wdl, q, maia_ldw, maia_top, maia_ent, fen, E, gp, view="root"):
    """Corrected (W, D, L) for the side to move of the evaluated position (fen). wdl: the engine's (W, D, L) for that side; q: the engine's
    value (pseudo-Leela Q from Stockfish) for that side; maia_ldw: Maia's [loss, draw, win] for that side; maia_top / maia_ent: Maia's
    policy stats of that position (same pass); E: the player's strength (single rating); gp: group_params(...); view: 'root' or 'after'.
    Positions outside the domain (more than 12 men, castling rights) are returned unchanged."""
    if gp is None or not applies(fen):
        return tuple(wdl)
    p = gp[view]
    W, D, L = [max(v, 1e-300) for v in wdl]
    t = math.log(W) - math.log(L)
    lnu = math.log(D) - (math.log(W) + math.log(L)) / 2
    f = features(q, maia_ldw, maia_top, maia_ent, fen, E, p)
    t += sum(a * f[k] for k, a in p["odd"].items())
    lnu += sum(b * f[k] for k, b in p["even"].items())
    m = max(t / 2, -t / 2, lnu)
    ew, el, en = math.exp(t / 2 - m), math.exp(-t / 2 - m), math.exp(lnu - m)
    zz = ew + el + en
    return ew / zz, en / zz, el / zz
