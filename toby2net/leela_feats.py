# -*- coding: utf-8 -*-
"""Leela search inputs for B (pilot, Yosha 2026-11-17): Toby 1.0 Accurate's main search, as the engine runs it (toby_uci.Lc0().search(fen,
1024): lc0 0.32, BT3-768x15x24h-swa-2790000, cuda backend, --backend-opts=max_batch=256, --verbose-move-stats, position given as FEN only,
tree reset before each search). Accurate's 64-node complements of the unvisited moves are not run: those moves are marked unvisited.
Per legal move (the order of the shard's `lm`): lshare = visits / visits of all root moves, lq = the move's Q from the verbose stats, lvis
= visited (1 / 0); per position: lroot = the root's WL, lhas = 1 (search done; 0 = failed, the row keeps masked Leela inputs and is counted).
  python leela_feats.py IN.npz... --out DIR [--procs N]   -> DIR/<name>.npz = the input shard + these fields (resumable by file)
The engine (toby2.py, Toby 2 Accurate) computes its inputs with the same two functions, `lc0_fen` and `move_features`, on the same
toby_uci.Lc0 search: one code path for the head's training features and for the engine (lesson P1).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

NODES = 1024
FIELDS = ("lshare", "lq", "lvis", "lroot", "lhas")   # per move: visit share, Q, visited; per position: root WL, search present
_L = None


def lc0_fen(board):
    """The FEN given to lc0: the position alone, as rebuilt from a training row (toby_values.board_of): pieces, side to move, castling,
    en passant square only when a legal capture needs it (python-chess's default), rule-50 count, and move number 1 (a row has none)."""
    f = board.fen().split()
    return " ".join(f[:5] + ["1"])


def move_features(moves, root, legal):
    """toby_uci.Lc0.search's result ({uci: (visits, WL, D, Q)}, (WL, D) of the root) -> (share, q, vis) per move of `legal` (lists, in
    that order) and the root's WL. Unvisited or unlisted moves: share 0, Q 0, visited 0."""
    tot = sum(v[0] for v in moves.values()) or 1
    share = [moves[u][0] / tot if u in moves else 0.0 for u in legal]
    q = [moves[u][3] if u in moves else 0.0 for u in legal]
    vis = [1 if (u in moves and moves[u][0] > 0) else 0 for u in legal]
    return share, q, vis, float(root[0]) if root else 0.0


def lc0():
    global _L
    if _L is None:
        import toby_uci as T
        _L = T.Lc0()
    return _L


def one(job):
    i, r = job
    z = {"bb": r["bb"][None, None], "turn": np.array([r["turn"]]), "castle": r["castle"][None], "r50": np.array([r["r50"]]),
         "lm": r["lm"], "off": np.array([0, len(r["lm"])])}
    import toby_values as TV
    b, legal = TV.board_of(z, 0)
    try:
        moves, root = lc0().search(lc0_fen(b), NODES)
    except Exception as exc:  # noqa: BLE001
        return i, None, f"search failed: {type(exc).__name__}"
    return i, move_features(moves, root, legal), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N rows of each shard only (benchmark)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, done, fail = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for f in a.src:
            dst = out / Path(f).name
            if dst.exists() and not a.limit:
                continue
            z = dict(np.load(f))
            n = len(z["tgt"]) if not a.limit else min(a.limit, len(z["tgt"]))
            part = out / f"partial_{Path(f).name}"   # incremental copy (pulled off the pod as it grows): a watchdog never holds the only copy
            res = {}
            if part.exists():
                pz = np.load(part)
                cut = np.concatenate([[0], np.cumsum(pz["lens"])])
                for j, i in enumerate(pz["idx"]):
                    lo, hi = cut[j], cut[j + 1]
                    res[int(i)] = (pz["share"][lo:hi], pz["q"][lo:hi], pz["vis"][lo:hi], float(pz["root"][j]))
                print(f"resumed {len(res)} positions from {part}", flush=True)

            def save_partial():
                ks = sorted(res)
                tmp = str(part) + ".tmp.npz"
                np.savez(tmp, idx=np.array(ks, np.int64), lens=np.array([len(res[k][0]) for k in ks], np.int64),
                         share=np.concatenate([np.asarray(res[k][0], np.float16) for k in ks]) if ks else np.zeros(0, np.float16),
                         q=np.concatenate([np.asarray(res[k][1], np.float16) for k in ks]) if ks else np.zeros(0, np.float16),
                         vis=np.concatenate([np.asarray(res[k][2], np.uint8) for k in ks]) if ks else np.zeros(0, np.uint8),
                         root=np.array([res[k][3] for k in ks], np.float16))
                os.replace(tmp, part)
            jobs = [(i, {"bb": z["bb"][i, 0], "turn": int(z["turn"][i]), "castle": z["castle"][i], "r50": int(z["r50"][i]),
                         "lm": z["lm"][z["off"][i]:z["off"][i + 1]]}) for i in range(n) if i not in res]
            for i, r, err in pool.imap_unordered(one, jobs, chunksize=4):
                done += 1
                if err:
                    fail += 1
                else:
                    res[i] = r
                if done >= 200 and fail > 0.01 * done:   # lc0 broken (missing network / binary / GPU): stop, never write an output
                    print(f"STOP: {fail} of {done} searches failed", flush=True)
                    pool.terminate()
                    sys.exit(3)
                if done % 2000 == 0:
                    print(f"  {done} positions, {done / (time.time() - t0):.1f} pos/s, failed {fail}", flush=True)
                    if not a.limit:
                        save_partial()
            L = len(z["lm"])
            sh, q, vis = np.zeros(L, np.float16), np.zeros(L, np.float16), np.zeros(L, np.uint8)
            root, has = np.zeros(len(z["tgt"]), np.float16), np.zeros(len(z["tgt"]), np.uint8)
            for i, (s_, q_, v_, r_) in res.items():
                lo, hi = z["off"][i], z["off"][i + 1]
                sh[lo:hi], q[lo:hi], vis[lo:hi], root[i], has[i] = s_, q_, v_, r_, 1
            z.update(lshare=sh, lq=q, lvis=vis, lroot=root, lhas=has,
                     lproto=np.array(f"toby_uci.Lc0().search(fen, 1024): lc0 0.32, BT3-768x15x24h-swa-2790000, backend "
                                     f"{os.environ.get('TOBY_BACKEND') or 'cuda'}, max_batch=256; no 64-node complements (unvisited moves masked)"),
                     lbackend=np.array(os.environ.get("TOBY_BACKEND") or "cuda"))
            if a.limit:
                print(f"BENCH {done} positions in {time.time() - t0:.0f} s: {done / (time.time() - t0):.2f} pos/s with {a.procs} lc0, "
                      f"failed {fail}", flush=True)
                return
            tmp = str(dst) + ".tmp.npz"
            np.savez_compressed(tmp, **z)
            os.replace(tmp, dst)
            print(f"SHARD {dst.name} {len(res)} / {n}, total {done}, {done / (time.time() - t0):.1f} pos/s, failed {fail}", flush=True)
    print(f"LEELA_DONE {done} positions, failed {fail}, {done / max(1, time.time() - t0):.1f} pos/s", flush=True)


if __name__ == "__main__":
    main()
