# -*- coding: utf-8 -*-
"""toby2.py - Toby 2's move model at engine time: network B (Leela BT4 body fine-tuned on human games, a move head and a W/D/L head)
with Stockfish 19 values computed by Toby 1.0's recipe (MultiPV 150k nodes on every legal move, then 250k on the candidates).
Part of Toby (AGPL v3, see LICENSE). Network B is a modified Leela Chess Zero BT4 network (GPL-3.0-or-later); its architecture and
input encoding are Leela Chess Zero's (GPL v3): see NOTICE.md.

The network is the training code's own `model.BodyB` (nn/v7/model.py, nn/v7/leela_torch.py, shipped next to this file in toby2net/),
split in two at engine time:
  body(planes, ratings / time control / clocks / venue)  -> (1, 64, E)   the 15 encoder layers; values never enter it
  heads(x, legal moves, value features)                  -> move logits over the legal moves, W/D/L logits (loss, draw, win)
so that one position costs ONE body pass, run at t = 0 in parallel with Stockfish's base search, and one or two head passes:
  t = 0       body pass (torch threads)  ||  Stockfish pass 1: MultiPV over every legal move, V2_BASE nodes (single thread, hash cleared)
  candidates  "Maia C": Toby 1.0's fine-tuned Maia's top 3 (known before t = 0, as Toby 1.0) | "Network": B's own top 3 from a head pass
              with every value input masked (as the masked-values training rows) -> the extension on them starts on a second Stockfish
              as soon as they are known; candidates = top 3 + Stockfish's top 2 at pass 1 (toby_uci.v2_candidates, same order)
  pass 2      V2_EXT nodes, MultiPV over the candidates (hash cleared); their values replace pass 1's
  heads       value features (model.value_features: the CLAUDE.md expected score at the mover's strength) -> distribution + W/D/L
The Stockfish calls are exactly toby_uci.sf_v2's (same searches, same candidate order): the values are those of the training data
whatever the timing.

Toby 2 Accurate (`extra_move_inputs`, the Leela head): a second, small file `toby2_lhead.pt` (format HEAD_FORMAT) holds the head trained
on the frozen B (`lpath` + its own value path `vpath`, nn/v7/pod/job_lhead_final.sh) and names the B file it belongs to (sha256). Per move
of the Accurate product, lc0 0.32 + BT3 searches 1,024 nodes from the FEN (nn/v7/leela_feats.py: `lc0_fen`, `move_features`, the same
functions as the head's training features), in parallel with the body and Stockfish; the 5 Leela inputs per move (visit share, Q,
visited, root WL, present) enter the head: logits = B's grid entry + B's per-move MLP + the head's value path + lpath([value features,
Leela inputs, both ratings]) (model.BodyB.forward with lfeat). Without the head (rapid / blitz by default, or when Leela cannot run) the
answer is B's own (B's value path). lc0 runs only on an Ampere sm_86 card (A40, RTX A6000 / A5000, RTX 3090: the card the head's inputs
were computed on); any other card, a missing lc0 or network, or a failed search is said and recorded in the path line, and that move is
answered by B with Stockfish values (never silently).

Weights: one self-contained file `toby2_b.pt` (format FORMAT, written by `python toby2.py export`): the architecture as recorded from
lc0's ONNX export of BT4, every tensor of the trained BodyB (fp16 where exact), the training option and the input format. Changing the
network = replacing that file (and re-running `export` / `onnx` from the new model_*.pt); nothing else changes.
Runtimes: "torch" (CPU fp32, or CUDA fp32 / fp16) and "onnx" (onnxruntime CPU, fp32 or dynamic int8; `python toby2.py onnx`).

Product configuration (`toby2_fast.json`, `toby2_accurate.json`, nn/v7/make_toby2_configs.py): every setting the sealed protocol fixes
(network files and their sha256, device and compute path, candidate source, values rule, clock rule, Leela settings, head per cadence);
`load_config` checks the hashes and the engine refuses a file that does not match.

Command line (development / build):
  export MODEL.pt --onnx BT4.onnx --pb BT4.pb.gz --out toby2_b.pt [--option C] [--vpath] [--freeze 0] [--run-meta log.jsonl]
  export-head HEAD_MODEL.pt --base-model B_MODEL.pt --base toby2_b.pt --out toby2_lhead.pt [--backend cuda-fp16] [--run-meta log.jsonl]
  onnx   toby2_b.pt --out DIR [--int8]      -> DIR/toby2_b.body.onnx, DIR/toby2_b.heads.onnx (+ .int8 body)
  info   toby2_b.pt"""
import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
for _d in (HERE / "toby2net", HERE / "nn" / "v7"):
    if (_d / "leela_torch.py").exists():
        sys.path.insert(0, str(_d))
        break
LT = EN = None   # leela_torch / encode (nn/v7, the training code's own modules): imported at the first use (torch is slow to import; the
                 # engine must answer 'uci' at once)


def _load_mods():
    global LT, EN
    if LT is None:
        import leela_torch as _lt
        import encode as _en
        LT, EN = _lt, _en
    return LT

FORMAT = "toby2-b/1"
HEAD_FORMAT = "toby2-lhead/1"
CONFIG_FORMAT = "toby2-config/1"
NL = 5          # Leela inputs per move (model.NL with V7_LEELA=1): visit share, Q, visited, root WL, present
LEELA_NODES = 1024
LEELA_BACKEND = "cuda-fp16"
# Which cards run the head's lc0: no allow-list of compute capabilities (an sm_86-only list was a property of our Linux pod build; the
# official Windows CUDA build has sm_89 kernels): every card must pass lc0_selfcheck (3 fixed positions against the stored A40 features).
NV = 4352
NF, NS, NSIDE = 7, 6, 12
CADENCES = ("bullet", "blitz", "rapid", "classical")   # nn/v7/common.py (training indices)
VENUES = ("online", "otb")
TIME_CONTROL_CADENCE = {"Classical": "classical", "Rapid": "rapid", "Blitz": "blitz", "Bullet": "bullet"}
CANDIDATE_SOURCES = ("Network", "Maia C")
# How "Network" takes B's top 3 (the product file's `candidate_pass`):
#   "masked"    B's heads with every value input masked, right after the body pass (sealed protocol 1.4 / gate 9 text: "B's own top 3
#               from a values-masked head pass"); the 250k extension starts at once on the second Stockfish, in parallel with pass 1
#   "base150k"  B's heads on the pass-1 values (150k nodes, every legal move), after pass 1; the extension then runs after it
CANDIDATE_PASSES = ("masked", "base150k")
SEARCH_SWITCH_INTERVAL = float(os.environ.get("TOBY2_SWITCH_INTERVAL", "0.0005"))   # Python's GIL switch interval while B's body / masked
# pass run next to the Stockfish reader threads (default 0.005 s elsewhere); scheduling only, the numbers are unchanged
_E_box = (2000.0, 2000.0)   # (mover, opponent) strengths for a maia_top3 callback (set by the caller before run)
# Toby 2's per-move W/D/L ("Move Evaluations = Toby 2", 2.0.2): which moves of a position get their child W/D/L (child_set)
CHILD_MASS = float(os.environ.get("TOBY2_CHILD_MASS", "0.99"))   # the most probable moves until this much probability is covered,
CHILD_MIN = int(os.environ.get("TOBY2_CHILD_MIN", "3"))          # at least this many (or every legal move when fewer), plus the extra moves
CHILD_BATCH = 32   # children per body pass (memory bound only)


def _vocab():
    names = [chess.square_name(a) + chess.square_name(b) for a in range(64) for b in range(64)]
    names += [f"{f}7{t}8{p}" for f in "abcdefgh" for t in "abcdefgh" for p in "qrbn"]
    return {m: i for i, m in enumerate(names)}


VOCAB = _vocab()   # = nn/v7/positions.VOCAB (mover's frame, mirrored for Black)


def _mirror(u):
    return u[0] + str(9 - int(u[1])) + u[2] + str(9 - int(u[3])) + u[4:]


def vidx(u, white):
    return VOCAB.get(u if white else _mirror(u), -1)


def default_weights():
    env = os.environ.get("TOBY2_WEIGHTS")
    if env:
        return Path(env)
    for c in (HERE / "nets" / "toby2_b.pt", HERE / "toby2_b.pt", HERE / "nn" / "toby2_b.pt"):
        if c.exists():
            return c
    return HERE / "nets" / "toby2_b.pt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ architecture without the ONNX file
_MISSING = object()


class _InitRec:
    """Records which constants of the ONNX export Lc0Net reads (small ones by value, large ones by shape only)."""

    def __init__(self, init, rec):
        self.i, self.rec = init, rec

    def _note(self, k):
        if k in self.i:
            v = np.asarray(self.i[k])
            self.rec[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "value": v.reshape(-1).tolist() if v.size <= 10000 else None}
        else:
            self.rec[k] = None

    def __contains__(self, k):
        self._note(k)
        return k in self.i

    def __getitem__(self, k):
        self._note(k)
        return self.i[k]

    def get(self, k, d=None):
        self._note(k)
        return self.i.get(k, d)


class ArchRecorder:
    """Wraps leela_torch.OnnxWeights: Lc0Net is built through it once (export), and every query is written down, so that the same network
    can later be built from the record alone (ArchReplay) and filled from the state dict."""

    def __init__(self, W):
        self.W = W
        self.arch = {"get": {}, "ln": {}, "act": {}, "init": {}}
        self.arch["encoders"] = sorted({n.split("/")[1] for n in W.nodes if n.startswith("/encoder")}, key=lambda s: int(s[7:]))
        self.nodes = {f"/{e}/": None for e in self.arch["encoders"]}
        self.init = _InitRec(W.init, self.arch["init"])

    def get(self, name):
        v = self.W.get(name)
        self.arch["get"][name] = None if v is None else list(np.asarray(v).shape)
        return v

    def ln(self, name):
        g, b = self.W.ln(name)
        self.arch["ln"][name] = [list(np.asarray(g).shape), list(np.asarray(b).shape)]
        return g, b

    def act_of(self, prefix):
        a = self.W.act_of(prefix)
        self.arch["act"][prefix] = a
        return a


class _InitReplay:
    def __init__(self, rec):
        self.rec = rec

    def _val(self, k):
        r = self.rec[k]
        if r is None:
            return None
        if r["value"] is not None:
            return np.array(r["value"], dtype=r["dtype"]).reshape(r["shape"])
        return np.zeros(r["shape"], np.float32)   # a parameter (filled from the state dict)

    def __contains__(self, k):
        return self.rec.get(k, _MISSING) not in (None, _MISSING)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self._val(k)

    def get(self, k, d=None):
        if self.rec.get(k, _MISSING) is _MISSING:
            raise KeyError(f"{k}: not in the recorded architecture")
        v = self._val(k)
        return d if v is None else v


class ArchReplay:
    """The recorded queries, answered with zero tensors of the recorded shapes (filled by load_state_dict) and the recorded constants."""

    def __init__(self, arch):
        self.a = arch
        self.nodes = {f"/{e}/": None for e in arch["encoders"]}
        self.init = _InitReplay(arch["init"])

    def get(self, name):
        s = self.a["get"].get(name, _MISSING)
        if s is _MISSING:
            raise KeyError(f"{name}: not in the recorded architecture")
        return None if s is None else np.zeros(s, np.float32)

    def ln(self, name):
        s = self.a["ln"][name]
        return np.zeros(s[0], np.float32), np.zeros(s[1], np.float32)

    def act_of(self, prefix):
        return self.a["act"][prefix]


def _model_module():
    import model as M   # nn/v7/model.py (toby2net/model.py in the package)
    return M


def build_net(arch, info, vpath, freeze):
    _load_mods()
    M = _model_module()
    lc0 = LT.Lc0Net(ArchReplay(arch) if not isinstance(arch, (ArchRecorder, LT.OnnxWeights)) else arch, info)
    net = M.BodyB(lc0, freeze=freeze, vpath=vpath)
    net.lpath = None   # B's file never holds the Leela head (whatever V7_LEELA says): the head comes from its own file (head_modules)
    return net


# ------------------------------------------------------------------------------------------------ export (development / build)
def export(model_pt, onnx_path, pb_path, out, option="C", vpath=True, freeze=0, run_meta=None):
    """Training weights (train.py model_*.pt, fp16 state dict of BodyB) -> one self-contained engine file. The network is built exactly as
    train.load_body does (lc0's ONNX export + the state dict, strict=False), then every tensor of the resulting module is stored."""
    import torch
    _load_mods()
    info = LT.pb_info(pb_path)
    info = {k: (v if isinstance(v, (int, float, str)) else bytes(v).hex()) for k, v in info.items()}
    rec = ArchRecorder(LT.OnnxWeights(onnx_path))
    net = build_net(rec, info, vpath, freeze)
    sd = torch.load(model_pt, map_location="cpu")
    missing, unexpected = net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in {model_pt}: {unexpected[:5]}")
    state, exact16, shared = {}, 0, {}
    for k, v in net.state_dict().items():   # tensors shared by several modules (smolgen's weight, read by every encoder) stored once
        key = (v.data_ptr(), tuple(v.shape), v.dtype)
        if key in shared:
            state[k] = shared[key]
            continue
        if v.is_floating_point() and torch.equal(v.half().float(), v):
            state[k] = v.half()
            exact16 += 1
        else:
            state[k] = v.clone()
        shared[key] = state[k]
    meta = {"format": FORMAT, "option": option, "vpath": bool(vpath), "freeze": int(freeze), "lc0_input_format": int(info.get("input", 0)),
            "source_model": Path(model_pt).name, "source_model_sha256": sha256(model_pt), "bt4_onnx_sha256": sha256(onnx_path),
            "bt4_pb_sha256": sha256(pb_path), "bt4_licence": info.get("license", ""), "keys_from_bt4_only": list(missing),
            "tensors": len(state), "fp16_exact": exact16, "extra_move_inputs": 0,
            "params": int(sum(v.numel() for k, v in state.items() if v.is_floating_point() and not k.endswith(".smol_w.w") or k == "lc0.smol.w"))}
    if run_meta and Path(run_meta).exists():   # the training run's own metadata (first line of log.jsonl or summary json)
        try:
            first = json.loads(Path(run_meta).read_text().splitlines()[0])
            meta["training"] = {k: first[k] for k in ("body", "option", "vpath", "freeze", "init", "data2", "vweight", "seed") if k in first}
        except Exception:  # noqa: BLE001
            pass
    torch.save({"meta": meta, "arch": rec.arch, "info": info, "state": state}, out)
    print(f"EXPORT {out}: {len(state)} tensors ({exact16} stored fp16), {Path(out).stat().st_size / 1e6:.0f} MB, missing from the "
          f"training file (BT4 values kept): {len(missing)}", flush=True)
    return meta


def head_modules():
    """The Leela head's two small MLPs exactly as model.BodyB builds them (vpath: NF + 24 inputs; lpath: NF + NL + 24), loaded strictly."""
    import torch.nn as nn
    mlp = lambda n: nn.Sequential(nn.Linear(n, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    return nn.ModuleDict({"vpath": mlp(NF + 24), "lpath": mlp(NF + NL + 24)})


def export_head(head_pt, base_model_pt, base_engine, out, backend=LEELA_BACKEND, run_meta=None):
    """Leela-head training weights (train.py --train-only lpath,vpath on a frozen B: a full BodyB state dict) -> toby2_lhead.pt with the
    head's tensors only. Every other tensor must equal B's (the body was frozen): checked here, so that the head file plus the B file are
    exactly the trained T2A network."""
    import torch
    sd = torch.load(head_pt, map_location="cpu")
    sb = torch.load(base_model_pt, map_location="cpu")
    head = {k: v for k, v in sd.items() if k.startswith(("lpath.", "vpath."))}
    if not any(k.startswith("lpath.") for k in head):
        raise SystemExit(f"{head_pt}: no lpath tensors (not a Leela-head run)")
    diff = [k for k, v in sd.items() if k not in head and (k not in sb or not torch.equal(v, sb[k]))]
    if diff:
        raise SystemExit(f"{head_pt}: {len(diff)} tensors outside the head differ from {base_model_pt} (B not frozen?): {diff[:5]}")
    bmeta = torch.load(base_engine, map_location="cpu", weights_only=False)["meta"]
    if bmeta.get("source_model_sha256") != sha256(base_model_pt):
        raise SystemExit(f"{base_engine} was not exported from {base_model_pt}")
    mods = head_modules()
    mods.load_state_dict({k: v.float() for k, v in head.items()}, strict=True)
    state = {}
    for k, v in mods.state_dict().items():
        state[k] = v.half() if torch.equal(v.half().float(), v) else v.clone()
    meta = {"format": HEAD_FORMAT, "extra_move_inputs": NL, "leela_fields": ["lshare", "lq", "lvis", "lroot", "lhas"],
            "base_engine_sha256": sha256(base_engine), "base_source_model_sha256": bmeta["source_model_sha256"],
            "source_model": str(head_pt), "source_model_sha256": sha256(head_pt), "tensors_checked_equal_to_B": len(sd) - len(head),
            "leela": {"lc0": "0.32", "network": "BT3-768x15x24h-swa-2790000", "nodes": LEELA_NODES, "backend": backend,
                      "position": "FEN only (leela_feats.lc0_fen), tree reset", "complements": "none (unvisited moves masked)"},
            "fp16_exact": sum(1 for v in state.values() if v.dtype == torch.float16)}
    if run_meta and Path(run_meta).exists():
        try:
            first = json.loads(Path(run_meta).read_text().splitlines()[0])
            meta["training"] = {k: first[k] for k in ("body", "option", "vpath", "init", "train_only", "seed", "data") if k in first}
        except Exception:  # noqa: BLE001
            pass
    torch.save({"meta": meta, "state": state}, out)
    print(f"EXPORT-HEAD {out}: {len(state)} tensors ({meta['fp16_exact']} fp16-exact), {Path(out).stat().st_size / 1e3:.0f} kB; "
          f"{meta['tensors_checked_equal_to_B']} other tensors equal to B", flush=True)
    return meta


# ------------------------------------------------------------------------------------------------ the Leela search inputs (Accurate)
_GPU_PROBE = []   # the decision, cached for the process (probed once)


def gpu_for_lc0(timeout=5.0):
    """(ok, text): the NVIDIA card lc0 would run on (name and compute capability), from nvidia-smi (lc0 does not go through PyTorch), at
    most `timeout` seconds, once per process (cached). CUDA_VISIBLE_DEVICES empty, -1 or naming no listed card = no GPU visible. Whether
    the card is good enough is decided by lc0_selfcheck, not here (lesson I1: a card is accepted on evidence, never assumed)."""
    if _GPU_PROBE:
        return _GPU_PROBE[0]
    _GPU_PROBE.append(_gpu_probe(timeout))
    return _GPU_PROBE[0]


NVIDIA_SMI_TIMEOUT = float(os.environ.get("TOBY_NVIDIA_SMI_TIMEOUT", "5"))   # hard wall-clock limit of the one nvidia-smi call
_SMI = {}   # nvidia-smi's answers in this process, per argument list: (returncode or None, stdout text, error text or None)
GPU_QUERY = ("--query-gpu=name,compute_cap", "--format=csv,noheader")
_SMI_LOCK = threading.Lock()


def _kill_tree(proc):
    """Stops a child and everything it started: on Windows `taskkill /T /F` (a .bat wrapper's cmd.exe and its children), elsewhere its
    process group (it was started in a new session)."""
    import subprocess
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def nvidia_smi(args=GPU_QUERY, timeout=None):
    """The process's ONE call of `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` (cached; every caller gets the same
    answer): (returncode, stdout, None), or (None, "", reason) when nvidia-smi is missing, fails to start or does not finish within the
    hard wall-clock limit. Its output goes to a temporary file, not a pipe, and on time-out the whole process tree is killed: a wrapper
    (a .bat on Windows) that hangs while holding a pipe cannot block the engine."""
    with _SMI_LOCK:
        if args in _SMI:
            return _SMI[args]
        import shutil
        import subprocess
        import tempfile
        limit = NVIDIA_SMI_TIMEOUT if timeout is None else float(timeout)
        exe = shutil.which("nvidia-smi")
        if not exe:
            _SMI[args] = (None, "", "no nvidia-smi (no NVIDIA driver)")
            return _SMI[args]
        res = None
        try:
            with tempfile.TemporaryFile() as fo:
                kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)} \
                    if os.name == "nt" else {"start_new_session": True}
                proc = subprocess.Popen([exe, *args], stdin=subprocess.DEVNULL, stdout=fo,
                                        stderr=subprocess.DEVNULL, **kw)
                end = time.time() + limit
                while proc.poll() is None and time.time() < end:
                    time.sleep(0.02)
                if proc.poll() is None:
                    _kill_tree(proc)
                    res = (None, "", f"nvidia-smi did not answer within {limit:.0f} s")
                else:
                    fo.seek(0)
                    res = (proc.returncode, fo.read().decode("utf-8", "replace"), None)
        except Exception as exc:  # noqa: BLE001
            res = (None, "", f"nvidia-smi failed ({type(exc).__name__})")
        _SMI[args] = res
        return res


def nvidia_rows():
    """[(name, compute capability), ...] of the cards nvidia-smi lists (its one cached call), [] when it lists none or failed."""
    rc, txt, err = nvidia_smi()
    if err or rc != 0:
        return []
    return [tuple(x.strip() for x in l.split(",")[:2]) for l in txt.strip().splitlines() if l.strip() and len(l.split(",")) >= 2]


def _gpu_probe(timeout):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        return False, f"no GPU visible (CUDA_VISIBLE_DEVICES={cvd!r})"
    rc, txt, err = nvidia_smi()
    if err:
        return False, err
    rows = [l.split(",") for l in txt.strip().splitlines() if l.strip()]
    if rc != 0 or not rows or len(rows[0]) < 2:
        return False, f"nvidia-smi failed or lists no GPU (exit code {rc})"
    first = (cvd or "0").split(",")[0].strip()
    if not first.isdigit() or int(first) >= len(rows):
        return False, f"CUDA_VISIBLE_DEVICES={cvd!r} names no listed card"
    name, cc = (x.strip() for x in rows[int(first)][:2])
    return True, f"{name} (sm_{cc.replace('.', '')})"


def lc0_selfcheck(leela, doc):
    """(ok, text): lc0 on this card against the stored A40 training features of 3 fixed positions (doc = the product file's
    leela.selfcheck): on every position the search completes, its most visited move is the stored one, max |d visit share| over the legal
    moves <= tolerance and |d root WL| <= tolerance (A40 run-to-run noise on these positions: <= 0.002 share)."""
    tol = doc["tolerance"]
    worst, worst_r, t0 = 0.0, 0.0, time.time()
    for k, pos in enumerate(doc["positions"]):
        b = chess.Board(pos["fen"])
        legal = [m.uci() for m in b.legal_moves]
        lf, _ = leela(b, legal)
        ref = pos["visited"]
        sh = {u: float(lf[j, 0]) for j, u in enumerate(legal)}
        d = max(abs(sh[u] - (ref[u][0] if u in ref else 0.0)) for u in legal)
        dr = abs(float(lf[0, 3]) - pos["root_wl"])
        top, rtop = max(sh, key=sh.get), max(ref, key=lambda u: ref[u][0])
        worst, worst_r = max(worst, d), max(worst_r, dr)
        if d > tol["max_abs_dshare"] or dr > tol["max_abs_droot"] or (tol.get("top_visited_same") and top != rtop):
            return False, (f"self-check failed on position {k + 1}: max |d share| {d:.3f} (<= {tol['max_abs_dshare']}), |d root| {dr:.3f} "
                           f"(<= {tol['max_abs_droot']}), most visited {top} (stored {rtop})")
    return True, f"self-check passed ({len(doc['positions'])} positions, max |d share| {worst:.4f}, |d root| {worst_r:.4f}, {time.time() - t0:.1f}s)"


class LeelaInputs:
    """The head's per-move inputs at engine time: one lc0 search of LEELA_NODES nodes from leela_feats.lc0_fen(board) through
    toby_uci.Lc0 (the same class and settings as the training features), turned into features by leela_feats.move_features. lc0 is any
    object with search(fen, nodes) -> (moves, root) (toby_uci.Lc0). The fields are rounded to fp16 as the training shards store them."""

    def __init__(self, lc0, backend, nodes=LEELA_NODES):
        import leela_feats as LF
        self.LF, self.lc0, self.backend, self.nodes = LF, lc0, backend, nodes
        self.gpu = ""   # the card lc0 runs on (gpu_for_lc0's text), shown in the path line

    def __call__(self, board, legal):
        # (one legal move: lc0 lists no move statistics; the training features are then share / Q / visited 0 with the root's WL,
        # present = 1, and so are these: no special case, the same function)
        moves, root = self.lc0.search(self.LF.lc0_fen(board), self.nodes)
        share, q, vis, rwl = self.LF.move_features(moves, root, legal)
        f16 = lambda x: np.asarray(x, np.float16).astype(np.float32)
        L = len(legal)
        lf = np.zeros((L, NL), np.float32)
        lf[:, 0], lf[:, 1], lf[:, 2] = f16(share), f16(q), np.asarray(vis, np.float32)
        lf[:, 3], lf[:, 4] = f16([rwl])[0], 1.0
        return lf, {"visited": int(sum(vis)), "nodes": int(sum(v[0] for v in moves.values())), "root": root is not None}


# ------------------------------------------------------------------------------------------------ product configuration
def load_config(path):
    """A product file (toby2_fast.json / toby2_accurate.json): the dict, with the network files resolved next to it and their sha256
    checked. Raises on any mismatch: the engine then says so and Toby 1.0 answers (never a silent substitute)."""
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("format") != CONFIG_FORMAT:
        raise RuntimeError(f"{path}: not a Toby 2 product file ({CONFIG_FORMAT})")
    for key in ("network", "head"):
        ent = cfg.get(key)
        if not ent:
            continue
        f = Path(ent["file"])
        if not f.is_absolute():
            f = path.parent / f
        if not f.exists():
            raise RuntimeError(f"{path.name}: {key} file {f} missing")
        h = sha256(f)
        if h != ent["sha256"]:
            raise RuntimeError(f"{path.name}: {f.name} sha256 {h[:16]} differs from the product file's {ent['sha256'][:16]}")
        ent["path"] = str(f)
    cfg["config_sha256"] = sha256(path)
    cfg["config_file"] = path.name
    return cfg


# ------------------------------------------------------------------------------------------------ the network at engine time
class Toby2Net:
    """B at engine time. device 'cpu' | 'cuda'; dtype 'fp32' | 'fp16' (CUDA only); runtime 'torch' | 'onnx' (CPU, onnxruntime)."""

    def __init__(self, path=None, device="cpu", dtype="fp32", runtime="torch", threads=0, onnx_dir=None, int8=False, head=None):
        import torch
        _load_mods()
        self.torch = torch
        self.path = Path(path or default_weights())
        t0 = time.time()
        blob = torch.load(self.path, map_location="cpu", weights_only=False)
        if blob.get("meta", {}).get("format") != FORMAT:
            raise RuntimeError(f"{self.path}: not a Toby 2 weights file ({FORMAT})")
        self.meta, self.info = blob["meta"], blob["info"]
        if self.meta.get("extra_move_inputs", 0):
            raise RuntimeError("this network reads per-move Leela-search inputs: not supported by this engine")
        self.option = self.meta["option"]
        self.fmt = int(self.meta["lc0_input_format"])
        net = build_net(blob["arch"], self.info, self.meta["vpath"], self.meta["freeze"])
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in blob["state"].items()}, strict=True)
        del blob
        self.head = self.head_meta = None
        if head:   # Toby 2 Accurate: the Leela head of THIS B (checked by the B file's sha256), kept in fp32 on the network's device
            hb = torch.load(head, map_location="cpu", weights_only=False)
            hm = hb.get("meta", {})
            if hm.get("format") != HEAD_FORMAT or int(hm.get("extra_move_inputs", 0)) != NL:
                raise RuntimeError(f"{head}: not a Toby 2 Leela-head file ({HEAD_FORMAT}, {NL} inputs per move)")
            if hm.get("base_engine_sha256") != sha256(self.path):
                raise RuntimeError(f"{Path(head).name} belongs to another B file ({str(hm.get('base_engine_sha256'))[:16]}), not {self.path.name}")
            mods = head_modules()
            mods.load_state_dict({k: v.float() for k, v in hb["state"].items()}, strict=True)
            self.head = mods.eval().to(device)
            self.head_meta = hm
            self.head_path = Path(head)
        if threads:
            torch.set_num_threads(int(threads))
        self.device = device
        self.half = device.startswith("cuda") and dtype == "fp16"
        net = net.eval().to(device)
        if self.half:
            net = net.half()
        self.net = net
        self.M = _model_module()
        self.runtime = runtime
        self.ort = None
        if runtime == "onnx":
            self._ort_open(onnx_dir or self.path.parent, int8, threads)
        self.load_seconds = time.time() - t0
        self.file_sha256 = sha256(self.path)
        self.label = {"network": self.path.name, "sha256": self.file_sha256[:16], "source_model": self.meta.get("source_model"),
                      "runtime": runtime + (" int8" if (runtime == "onnx" and int8) else ""), "device": device,
                      "dtype": "fp16" if self.half else "fp32"}
        if self.head is not None:
            self.label["head_file"] = self.head_path.name
            self.label["head_sha256"] = sha256(self.head_path)[:16]

    # ---- encoding (the training data's functions: leela_torch.lc0_history / lc0_planes / planes_tensor, encode.side_inputs)
    def encode(self, board):
        """board: python-chess board with its move stack (from the GUI's position command) -> dict of the position's inputs."""
        root = board.root()
        moves = [m.uci() for m in board.move_stack]
        hist = LT.lc0_history(moves, board=root)
        masks, r50, t = LT.lc0_planes(hist, self.fmt)
        torch = self.torch
        m = torch.from_numpy(masks.astype(np.uint64).view(np.int64)).view(1, -1)
        black = board.turn == chess.BLACK
        planes = LT.planes_tensor(m, torch.tensor([r50]), torch.tensor([1 if black else 0]), self.fmt)
        legal = [mv.uci() for mv in board.legal_moves]
        mi = [vidx(u, not black) for u in legal]
        if min(mi) < 0:
            raise RuntimeError(f"move outside the vocabulary in {board.fen()}")
        return {"planes": planes, "transform": int(t), "legal": legal, "mi": torch.tensor(mi, dtype=torch.long), "black": black}

    @staticmethod
    def side_inputs(base, inc, clk, clko, cadence, venue):
        """(1, 12): encode.side_inputs (the training code's function: [clock known, f(base), f(inc), f(mover's clock), f(opponent's clock),
        mover's share of base, cadence x4, venue x2] x 10). Any of base / inc / clk / clko (seconds) may be None = unknown: the masked
        path, exactly as the training rows without clocks (NaN)."""
        import torch
        g = lambda v: torch.tensor([float("nan") if v is None else float(v)], dtype=torch.float32)
        return EN.side_inputs(g(base), g(inc), g(clk), g(clko), torch.tensor([CADENCES.index(cadence)]), torch.tensor([VENUES.index(venue)]))

    def extra(self, E, Eo, side):
        """(1, 36): both ratings through the RBF (model.rating_rbf) + the 12 side inputs, as BodyB.forward builds them."""
        torch = self.torch
        rr = torch.cat([self.M.rating_rbf(torch.tensor([float(E)])), self.M.rating_rbf(torch.tensor([float(Eo)]))], -1)
        return rr, torch.cat([rr, side.to(rr.dtype).view(1, -1)], -1)

    # ---- body
    def body(self, enc, E, Eo, side):
        torch = self.torch
        rr, extra = self.extra(E, Eo, side)
        if self.ort is not None:
            x = self.ort["body"].run(None, {"planes": enc["planes"].numpy().astype(np.float32), "extra": extra.numpy().astype(np.float32)})[0]
            return {"x": torch.from_numpy(x), "rr": rr}
        with torch.inference_mode():
            dt = torch.float16 if self.half else torch.float32
            p, e = enc["planes"].to(self.device, dt), extra.to(self.device, dt)
            net = self.net
            x = net.lc0.body(p, e, net.extra_lin, net.inject(e) if net.freeze > 0 else None, net.freeze)
        return {"x": x, "rr": rr}

    # ---- value features (model.value_features on the dense vocabulary, as the training code)
    def features(self, enc, values, E):
        """values: {uci: cp from the mover's view, mates +-10000} or None (masked: no move valued) -> (feats (L, NF), summ (1, NS))."""
        torch = self.torch
        mi = enc["mi"]
        legal = torch.zeros(1, NV, dtype=torch.bool)
        legal[0, mi] = True
        vals = torch.zeros(1, NV, 3)
        if values is None:
            valued = torch.zeros_like(legal)
        else:
            v = torch.tensor([float(values[u]) for u in enc["legal"]])
            vals[0, mi] = v.view(-1, 1).expand(-1, 3)   # Toby 1.0's recipe gives one value: the training shards repeat it in the 3 columns
            valued = legal
        use_md = {"C": False, "Cmd": True, "none": None}[self.option]
        feats, summ = self.M.value_features(vals, legal, valued, torch.tensor([float(E)]), "B", use_md)
        return feats[0, mi], summ

    def features_masked(self, enc, E):
        """features(enc, None, E) without building the (1, NV, 3) tensors: with no move valued, model.value_features gives zero
        per-move features and a constant position summary. The first call computes both with the real function (and checks the zeros);
        later calls return the same tensors. Output-identical, cheaper on the candidate pass's critical path."""
        torch = self.torch
        if getattr(self, "_masked_summ", None) is None:
            f, s_ = self.features(enc, None, E)
            if not bool((f == 0).all()):
                raise RuntimeError("masked value features are not all zero: the constant fast path does not apply")
            self._masked_summ = s_.clone()
            return f, s_
        return torch.zeros(len(enc["legal"]), NF), self._masked_summ.clone()

    # ---- children: Toby 2's W/D/L of the positions after the root's moves (2.0.2: "Move Evaluations = Toby 2")
    def planes_of(self, board):
        """The lc0 planes (1, 112, 8, 8) of a board with its move stack: encode()'s first steps, without the legal-move part."""
        _load_mods()
        root = board.root()
        moves = [m.uci() for m in board.move_stack]
        hist = LT.lc0_history(moves, board=root)
        masks, r50, _ = LT.lc0_planes(hist, self.fmt)
        torch = self.torch
        m = torch.from_numpy(masks.astype(np.uint64).view(np.int64)).view(1, -1)
        return LT.planes_tensor(m, torch.tensor([r50]), torch.tensor([1 if board.turn == chess.BLACK else 0]), self.fmt)

    def masked_summary(self):
        """The W/D/L head's Stockfish summary with no move valued (model.value_features on a row whose value inputs are all masked, as
        the training rows without Stockfish values): a constant, computed once with the training code's function."""
        if getattr(self, "_masked_wdl_summ", None) is None:
            torch = self.torch
            legal = torch.zeros(1, NV, dtype=torch.bool)
            legal[0, 0] = True
            use_md = {"C": False, "Cmd": True, "none": None}[self.option]
            _, summ = self.M.value_features(torch.zeros(1, NV, 3), legal, torch.zeros_like(legal), torch.tensor([2000.0]), "B", use_md)
            self._masked_wdl_summ = summ.clone()
        return self._masked_wdl_summ

    def child_wdl(self, board, moves, E, Eo, side):
        """Network B's W/D/L of the positions after `moves` (legal moves of `board`, none ending the game), for THEIR side to move (the
        root's opponent): {uci: (loss, draw, win)} of that side. The child's inputs are those B was trained on: the planes with the
        history including the move; E / Eo = the child mover's and its opponent's strengths (the root's Eo, E); side = the child's side
        inputs (cadence, venue, the clocks as at the root, seen from the child's mover); every value input MASKED: the values of the
        child's own moves are unknown without searching the child, and B saw rows either fully valued or fully masked (the training rows
        without Stockfish values), never a row with some moves valued. Only B's body and its W/D/L head run (no move head). One body pass
        per CHILD_BATCH children: batched, so a child's numbers may differ from its single pass by float rounding (W/D/L only)."""
        torch = self.torch
        _load_mods()
        if not moves:
            return {}
        _, extra = self.extra(E, Eo, side)
        summ = self.masked_summary()
        net = self.net
        res = {}
        for k in range(0, len(moves), CHILD_BATCH):
            part = moves[k:k + CHILD_BATCH]
            planes = []
            for u in part:
                c = board.copy()
                c.push(chess.Move.from_uci(u))
                planes.append(self.planes_of(c))
            P = torch.cat(planes, 0)
            n = len(part)
            with torch.inference_mode():
                if self.ort is not None:   # onnxruntime body (the exported graph has batch 1: one child at a time), torch W/D/L head
                    x = torch.cat([torch.from_numpy(self.ort["body"].run(None, {"planes": P[i:i + 1].numpy().astype(np.float32),
                                                                                 "extra": extra.numpy().astype(np.float32)})[0])
                                   for i in range(n)], 0)
                    dt = next(net.parameters()).dtype
                else:
                    dt = torch.float16 if self.half else torch.float32
                    e = extra.to(self.device, dt).expand(n, -1)
                    x = net.lc0.body(P.to(self.device, dt), e, net.extra_lin, net.inject(e) if net.freeze > 0 else None, net.freeze)
                x = x.to(self.device, dt)
                wdl = net.lc0.value(x, net.wdl_sf(summ.to(self.device, dt).expand(n, -1))).float()[:, [2, 1, 0]]
                pr = torch.nn.functional.softmax(wdl.cpu(), 1).numpy()
            for u, row in zip(part, pr):
                res[u] = (float(row[0]), float(row[1]), float(row[2]))
        return res

    # ---- heads (BodyB.forward restricted to the legal moves of one position)
    def heads(self, enc, bout, feats, summ, lfeat=None):
        """-> (log-probabilities over enc['legal'] as a numpy array, W/D/L probabilities (loss, draw, win) for the side to move).
        lfeat (L, NL) numpy: the Leela inputs of the legal moves -> the Leela head (T2A); None -> B alone."""
        if lfeat is not None and self.head is None:
            raise RuntimeError("Leela inputs given but no Leela-head file loaded")
        torch = self.torch
        F = torch.nn.functional
        net = self.net
        t = enc["transform"]
        idx = LT.castle_fix(net.vt[t:t + 1].cpu(), enc["planes"])[0, enc["mi"]]
        sq = net.sqt[t].cpu()[enc["mi"]]
        L = len(enc["legal"])
        fr = torch.cat([feats, bout["rr"].expand(L, -1)], -1)
        if self.ort is not None:
            o = self.ort["heads"].run(None, {"x": bout["x"].numpy(), "idx": idx.numpy(), "sqf": sq[:, 0].numpy(), "sqt": sq[:, 1].numpy(),
                                             "fr": fr.numpy().astype(np.float32), "summ": summ.numpy().astype(np.float32)})
            logits, wdl = torch.from_numpy(o[0]), torch.from_numpy(o[1])
        else:
            if lfeat is not None and self.ort is not None:
                raise RuntimeError("the Leela head runs with the torch runtime only")
            with torch.inference_mode():
                hd = None
                if lfeat is not None:
                    hd = (self.head, torch.as_tensor(lfeat, dtype=torch.float32).to(self.device))
                logits, wdl = heads_math(net, bout["x"], idx.to(self.device), sq.to(self.device), fr.to(self.device), summ.to(self.device), hd)
        lp = F.log_softmax(logits.float().cpu(), 0)
        return lp.numpy(), F.softmax(wdl.float().cpu().view(-1), 0).numpy()


def heads_math(net, x, idx, sq, fr, summ, head=None):
    """The move and W/D/L heads of model.BodyB.forward for one position: x (1, 64, E), idx (L,) policy-grid entries of the legal moves,
    sq (L, 2) their transformed from / to squares, fr (L, NF + 24) value features + both ratings, summ (1, NS). Same operations as the
    training code (logits = grid entry + per-move MLP (+ value path); W/D/L = Leela's value head + the Stockfish summary).
    head = (modules {vpath, lpath}, lfeat (L, NL)): the Leela head replaces B's value path and adds lpath([feats, lfeat, ratings]),
    as BodyB.forward with lfeat (fp32, as the training code feeds them)."""
    dt = x.dtype
    grid = net.lc0.policy_grid(x)                      # (1, 67 * 64)
    logits = grid[0, idx].float()
    add = net.vhead(x[0, sq[:, 0]], x[0, sq[:, 1]], fr.to(dt)).float()
    if head is not None:
        mods, lf = head
        f32 = fr.float()
        add = add + mods["vpath"](f32).squeeze(-1).float()
        add = add + mods["lpath"](torch_cat([f32[:, :NF], lf.float(), f32[:, NF:]])).squeeze(-1).float()
    elif net.vpath is not None:   # the training code feeds the value path in fp32 (its weights' dtype here)
        add = add + net.vpath(fr.to(net.vpath[0].weight.dtype)).squeeze(-1).float()
    logits = logits + add
    wdl = net.lc0.value(x, net.wdl_sf(summ.to(dt))).float()[:, [2, 1, 0]]
    return logits, wdl


def torch_cat(xs):
    import torch
    return torch.cat(xs, -1)


# ------------------------------------------------------------------------------------------------ onnxruntime (CPU)
def _ort_open(self, d, int8, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    stem = self.path.stem
    body = Path(d) / f"{stem}.body{'.int8' if int8 else ''}.onnx"
    heads = Path(d) / f"{stem}.heads.onnx"
    for f in (body, heads):
        if not f.exists():
            raise RuntimeError(f"{f} missing (python toby2.py onnx)")
    self.ort = {"body": ort.InferenceSession(str(body), so, providers=["CPUExecutionProvider"]),
                "heads": ort.InferenceSession(str(heads), so, providers=["CPUExecutionProvider"]), "int8": int8}


Toby2Net._ort_open = _ort_open


def onnx_export(weights, out_dir, int8=False):
    """Two ONNX graphs from the engine file: body (planes (1,112,8,8), extra (1,36)) -> x (1,64,E); heads (x, idx (L,), sqf, sqt (L,),
    fr (L, NF + 24), summ (1, NS)) -> logits (L,), wdl (1, 3). Optional dynamic int8 quantisation of the body (MatMul weights)."""
    import torch
    net = Toby2Net(weights)
    n = net.net.float().eval()

    class Body(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, planes, extra):
            return self.n.lc0.body(planes, extra, self.n.extra_lin, self.n.inject(extra) if self.n.freeze > 0 else None, self.n.freeze)

    class Heads(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x, idx, sqf, sqt, fr, summ):
            return heads_math(self.n, x, idx, torch.stack([sqf, sqt], 1), fr, summ)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(weights).stem
    b = chess.Board()
    enc = net.encode(b)
    side = Toby2Net.side_inputs(5400, 30, None, None, "classical", "otb")
    rr, extra = net.extra(2000, 2000, side)
    with torch.no_grad():
        x = Body(n)(enc["planes"], extra)
    L = len(enc["legal"])
    pb = out_dir / f"{stem}.body.onnx"
    kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
    torch.onnx.export(Body(n), (enc["planes"], extra), str(pb), input_names=["planes", "extra"], output_names=["x"], opset_version=17, **kw)
    ph = out_dir / f"{stem}.heads.onnx"
    args = (x, torch.zeros(L, dtype=torch.long), torch.zeros(L, dtype=torch.long), torch.zeros(L, dtype=torch.long), torch.zeros(L, NF + 24),
            torch.zeros(1, NS))
    torch.onnx.export(Heads(n), args, str(ph), input_names=["x", "idx", "sqf", "sqt", "fr", "summ"], output_names=["logits", "wdl"],
                      dynamic_axes={"idx": {0: "L"}, "sqf": {0: "L"}, "sqt": {0: "L"}, "fr": {0: "L"}, "logits": {0: "L"}}, opset_version=17,
                      **kw)
    print("ONNX", pb, f"{pb.stat().st_size / 1e6:.0f} MB", ph, f"{ph.stat().st_size / 1e6:.0f} MB", flush=True)
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        pq = out_dir / f"{stem}.body.int8.onnx"
        quantize_dynamic(str(pb), str(pq), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        print("ONNX int8", pq, f"{pq.stat().st_size / 1e6:.0f} MB", flush=True)


# ------------------------------------------------------------------------------------------------ one position: values + network
# ------------------------------------------------------------------------------------------------ blunder penalty (product-file field)
PENALTY_GROUP = {"classical": "classical", "rapid": "rapid_blitz", "blitz": "rapid_blitz"}   # bullet: not tuned, no penalty


def expected_score(R, x):
    """CLAUDE.md's expected score (points, 0-100) of a value x (centipawns from the mover's side; mates +-10000 not capped, other values
    capped at +-2000) at rating R: the judge's function (nn/sealed_t2/analyse_t2.es)."""
    if abs(x) < 9000:
        x = max(-2000.0, min(2000.0, float(x)))
    return 100 * (0.5 + 0.5 * (2 / (1 + math.exp((R * -0.00000274 + 0.00048) * x)) - 1))


def _pchip_slopes(x, y):
    """The node derivatives of a monotone cubic Hermite interpolant (Fritsch-Carlson, the same rule as scipy.interpolate.PchipInterpolator:
    weighted harmonic mean inside, the three-point shape-preserving formula at the ends). Pure numpy: the engine needs no scipy."""
    h = np.diff(x)
    m = np.diff(y) / h
    n = len(x)
    d = np.zeros(n)
    if n == 2:
        d[:] = m[0]
        return d
    for k in range(1, n - 1):
        if m[k - 1] * m[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
    for k, (h0, h1, m0, m1) in ((0, (h[0], h[1], m[0], m[1])), (n - 1, (h[-1], h[-2], m[-1], m[-2]))):
        e = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if np.sign(e) != np.sign(m0):
            e = 0.0
        elif np.sign(m0) != np.sign(m1) and abs(e) > abs(3 * m0):
            e = 3 * m0
        d[k] = e
    return d


def pchip_flat(knots_x, knots_y, E):
    """beta_E(E) of the 'knots' form: the monotone PCHIP through (knots_x, knots_y), flat below the first and beyond the last knot, >= 0
    (nn/pass3/fit_penalty.py beta_e; identity checked on a grid, nn/pass3/check_knots.py)."""
    x = np.asarray(knots_x, np.float64)
    y = np.asarray(knots_y, np.float64)
    e = min(max(float(E), x[0]), x[-1])
    k = min(int(np.searchsorted(x, e, side="right")) - 1, len(x) - 2)
    d = _pchip_slopes(x, y)
    h = x[k + 1] - x[k]
    t = (e - x[k]) / h
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return max(0.0, h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1])


def penalty_beta(spec, E, cadence):
    """beta(E) of the product file's `blunder_penalty` for this cadence; 0 if the cadence's group is not in the field. Two forms:
    - linear (K2, toby2_blunder_fix.md): max(0, b0 + b1 * (min(E, e_cap) - 2000) / 1000);
    - "knots" (pass 3, nn/PROTOCOL_TOBY2_PASS3_DRAFT.md sections 8-11): beta_E(E) = monotone PCHIP through (knots_E, knots_beta), flat
      outside the knots, times g = 1 (no phase factor)."""
    g = (spec or {}).get(PENALTY_GROUP.get(cadence, ""))
    if not g:
        return 0.0
    if g.get("form") == "knots":
        if g.get("g", 1) != 1:
            raise ValueError("blunder_penalty: only g = 1 is implemented for the knots form")
        return pchip_flat(g["knots_E"], g["knots_beta"], E)
    E = min(float(E), float(g.get("e_cap", 1e9)))   # e_cap: beta(E) = beta(e_cap) above it (no extrapolation past the fitted range)
    return max(0.0, float(g["b0"]) + float(g["b1"]) * (E - 2000.0) / 1000.0)


def blunder_penalty(logp, legal, values, E, cadence, spec):
    """The value-aware penalty of the move distribution (nn/reports/toby2_blunder_fix.md): p'(m) proportional to
    p(m) exp(-beta(E) max(0, L(m) - l0)), L(m) = ES(E, best value) - ES(E, v(m)) from the Stockfish values the engine used (every legal
    move); l0 = the group's `l0` (default 0). W/D/L is not touched.
    -> (log p' as a numpy array, beta). beta = 0 (or no values): the input array itself, unchanged (bit-identical answer)."""
    beta = penalty_beta(spec, E, cadence)
    if beta <= 0.0 or not values:
        return logp, 0.0
    l0 = float(spec[PENALTY_GROUP[cadence]].get("l0", 0.0))   # penalise only the loss beyond l0 (0 = every loss)
    es = np.array([expected_score(E, values[u]) for u in legal], np.float64)
    lq = np.asarray(logp, np.float64) - beta * np.maximum(0.0, es.max() - es - l0)
    lq -= lq.max()
    lq -= math.log(np.exp(lq).sum())
    return lq.astype(np.asarray(logp).dtype), beta


def top_moves(legal, lp, k=3):
    order = np.argsort(-lp, kind="stable")
    return [legal[i] for i in order[:k]]


def v2_candidates(top3, base):
    """toby_uci.v2_candidates: the source's top 3 + Stockfish's top 2 at the base budget, legal, no duplicate (same order)."""
    c = list(dict.fromkeys(list(top3) + sorted(base, key=base.get, reverse=True)[:2]))
    return [m for m in c if m in base]


def child_set(legal, logp, extra=(), mass=None, kmin=None):
    """The moves that get Toby 2's per-move W/D/L: the most probable moves (stable order of -log p, as top_moves) until their total
    probability reaches `mass` (CHILD_MASS, 0.99), at least `kmin` of them (CHILD_MIN, 3; every legal move when fewer), then each move
    of `extra` not already in (the engine adds its most probable move and Stockfish's best move)."""
    mass = CHILD_MASS if mass is None else mass
    kmin = CHILD_MIN if kmin is None else kmin
    lp = np.asarray(logp, np.float64)
    order = np.argsort(-lp, kind="stable")
    res, tot = [], 0.0
    for i in order:
        if tot >= mass and len(res) >= kmin:
            break
        res.append(legal[int(i)])
        tot += math.exp(float(lp[int(i)]))
    for u in extra:
        if u in legal and u not in res:
            res.append(u)
    return res


class Toby2Search:
    """Values + network for one position. eng / spec: toby_uci.StockfishCalc engines (single thread; spec may be None on one core);
    v2_base / v2_ext: node budgets (toby_uci.V2_BASE / V2_EXT). maia_top3: a function board -> Maia C's top 3 (candidate source "Maia C")."""

    def __init__(self, net, eng, spec, v2_base, v2_ext, maia_top3=None, spec_lock=None, cand_pass="masked", spec3=None):
        if cand_pass not in CANDIDATE_PASSES:
            raise ValueError(f"candidate pass {cand_pass}")
        self.cand_pass = cand_pass
        self.spec3 = spec3   # third single-thread Stockfish (optional): speculative complements on B's top 3 + pass 1's provisional top 2
        self.net, self.eng, self.spec = net, eng, spec
        self.v2_base, self.v2_ext = v2_base, v2_ext
        self.maia_top3 = maia_top3
        self.spec_lock = spec_lock or threading.Lock()   # toby_uci passes _SPEC["busy"]: the second engine is shared with review batches
        self.next_body = None   # (board, E, Eo, side, source) of the search that will follow this one (toby_uci's Deepening), or None
        self._prebody = None    # (key, thread, box) of a prefetched body pass

    @staticmethod
    def _body_key(board, E, Eo, side, source, cand_pass):
        """Everything the body thread's result depends on: the position with its history, both ratings, the side inputs, the candidates."""
        return (board.root().fen(), tuple(m.uci() for m in board.move_stack), float(E), float(Eo),
                np.asarray(side.detach().cpu().numpy(), np.float32).tobytes(), source, cand_pass)

    def prefetch_body(self, board, E, Eo, side, source="Network"):
        """Output-identical pipelining (Deepening in toby_uci: several positions in a row): the body thread's work for a LATER search
        (encoding, B's body, the masked candidate pass) runs now in a thread, while this search waits for its Stockfish complements. The
        later run() uses it only if its own inputs are exactly the same (_body_key); it then starts both Stockfish searches at t = 0.
        The network is deterministic for the same inputs: the same numbers as computing them in that run."""
        key = self._body_key(board, E, Eo, side, source, self.cand_pass)
        b2, box = board.copy(), {}

        def _run():
            try:
                te = time.time()
                enc = self.net.encode(b2)
                box["t_encode"] = time.time() - te
                tb = time.time()
                box["out"] = self.net.body(enc, E, Eo, side)
                box["t"] = time.time() - tb
                box["enc"] = enc
                if source == "Network" and self.cand_pass == "masked":
                    f0, s0 = self.net.features_masked(enc, E)
                    lp0, _ = self.net.heads(enc, box["out"], f0, s0)
                    box["masked"] = lp0
                    box["guess"] = top_moves(enc["legal"], lp0, 3)
            except Exception as exc:  # noqa: BLE001   (the later run() then computes everything itself)
                box["err"] = exc
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        self._prebody = (key, th, box)

    def _take_body(self, board, E, Eo, side, source):
        pb, self._prebody = self._prebody, None
        if pb is None or pb[0] != self._body_key(board, E, Eo, side, source, self.cand_pass):
            return None
        pb[1].join()
        box = pb[2]
        if "err" in box or "enc" not in box or (source == "Network" and self.cand_pass == "masked" and "guess" not in box):
            return None
        return box

    def run(self, board, E, Eo, side, source="Network", values=True, stop=None, leela=None, on_base=None, before_heads=None):
        """-> dict: legal, logp (numpy), wdl (loss, draw, win), values {uci: cp} or None, candidates, timings (s), masked_logp (source Network),
        lfeat (the Leela inputs used, or None). values=False: the network alone (time trouble, or no Stockfish): value inputs masked.
        leela: a LeelaInputs (Toby 2 Accurate with the head on): its search runs from t = 0 in parallel with the body and Stockfish; if
        it fails, the answer is B's alone and the path line says why (`leela`: 'failed: ...', `head`: 'off').
        on_base(enc, body_out, base, leela_get): called once pass 1 is done (before the complements are waited for), for work that
        only needs the pass-1 values (toby_uci's Move Evaluations prefetch); it must not block, and nothing it does changes this
        answer. leela_get() waits for the Leela search (at most 2 s) and returns its inputs, or None.
        before_heads(): called (main thread) right before the final heads pass; it may wait (toby_uci: a child W/D/L batch running with
        more torch threads on the processor ends first, so the heads always run with the network's own thread count)."""
        t0 = time.time()
        tm = {}
        legal = [m.uci() for m in board.legal_moves]   # = encode()'s order; the encoding itself runs in the body thread (Stockfish starts first)
        res = {"legal": legal, "source": source, "values": None, "candidates": None, "lfeat": None,
               "path": dict(self.net.label, candidates=source if values and self.eng is not None else "-",
                            values="Stockfish V2" if values and self.eng is not None else "masked")}   # every answer records its path (no silent fallback)
        if leela is not None and (not values or self.eng is None):
            res["path"].update(head="off", leela="not run: the head needs Stockfish values")
            leela = None
        lbox, lth = {}, None
        if leela is not None:
            def _leela():
                try:
                    tl = time.time()
                    lbox["lf"], lbox["info"] = leela(lboard, legal)
                    lbox["t"] = time.time() - tl
                except Exception as exc:  # noqa: BLE001
                    lbox["err"] = f"{type(exc).__name__}: {exc}"
            lboard = board.copy(stack=False)   # each thread reads its own copy of the position
            lth = threading.Thread(target=_leela, daemon=True)
            lth.start()
        if not values or self.eng is None:
            enc = self.net.encode(board)
            bout = self.net.body(enc, E, Eo, side)
            tm["body"] = time.time() - t0
            f, s = self.net.features(enc, None, E)
            res["logp"], res["wdl"] = self.net.heads(enc, bout, f, s)
            tm["total"] = time.time() - t0
            res["timings"] = tm
            return res
        guess, sres, th, body_box = None, {}, None, {}

        def start_spec(g):
            nonlocal th
            if self.spec is None or len(g) < 2 or not self.spec_lock.acquire(blocking=False):
                return None

            def _run():
                try:
                    ts = time.time()
                    r = self.spec.extend(board, g, self.v2_ext)
                    if r is not None:
                        sres["v"] = r
                    sres["t"] = time.time() - ts
                except Exception as exc:  # noqa: BLE001
                    sres["err"] = exc
                finally:
                    self.spec_lock.release()
            th = threading.Thread(target=_run, daemon=True)
            th.start()
            return th
        if source == "Maia C":
            if self.maia_top3 is None:
                raise RuntimeError("candidate source Maia C: the fine-tuned Maia is not loaded")
            guess = list(self.maia_top3(board))
            start_spec(guess)

        bboard = board.copy()
        s3 = {"lock": threading.Lock(), "last": None, "target": None, "th": None, "cancel": None, "box": None, "starts": 0}
        board3 = board.copy(stack=False)

        def maybe_spec3(depth=None, per_d=None):   # on_depth(d, {uci: cp}) of pass 1, or () from the body thread
            """Output-identical speculation: as soon as B's top 3 and a completed pass-1 iteration are both known, the candidate list those
            give (v2_candidates) is searched on the third engine when it differs from B's top 3 alone (the second engine's speculation); a
            newer iteration that changes the list restarts it. Used only if the final list is exactly that one (same search, same order,
            cleared hash, 1 thread: the same values as the main engine would give)."""
            if self.spec3 is None or source != "Network" or self.cand_pass != "masked":
                return
            with s3["lock"]:
                if per_d is not None:
                    s3["last"] = per_d
                g, last = body_box.get("guess"), s3["last"]
                if g is None or last is None:
                    return
                prov = v2_candidates(g, last)
                if len(prov) < 2 or prov == g or prov == s3["target"]:
                    return
                if s3["th"] is not None:
                    s3["cancel"].set()
                    s3["th"].join()
                ev, box = threading.Event(), {}

                def _r3():
                    try:
                        t3 = time.time()
                        box["v"] = self.spec3.extend(board3, prov, self.v2_ext, cancel=ev)
                        box["t"] = time.time() - t3
                    except Exception as exc:  # noqa: BLE001
                        box["err"] = exc
                s3.update(target=prov, cancel=ev, box=box, th=threading.Thread(target=_r3, daemon=True), starts=s3["starts"] + 1)
                s3["th"].start()

        pre = self._take_body(board, E, Eo, side, source)   # this search's body pass, already made during the previous one (or None)

        def _body():   # torch releases the GIL: the encoding and the body run while Stockfish searches (separate processes)
            try:
                if pre is not None:
                    body_box.update({k: pre[k] for k in ("enc", "t_encode", "out", "t") if k in pre})
                    body_box["prefetched"] = True
                    if source == "Network" and self.cand_pass == "masked":
                        body_box["masked"] = pre["masked"]
                        body_box["guess"] = pre["guess"]
                        start_spec(body_box["guess"])
                        maybe_spec3()
                    body_box["tc"] = time.time() - t0
                    return
                te = time.time()
                enc = body_box["enc"] = self.net.encode(bboard)
                body_box["t_encode"] = time.time() - te
                tb = time.time()
                body_box["out"] = self.net.body(enc, E, Eo, side)
                body_box["t"] = time.time() - tb
                if source == "Network" and self.cand_pass == "masked":
                    f0, s0 = self.net.features_masked(enc, E)   # constant masked features: the candidates come sooner, same numbers
                    lp0, _ = self.net.heads(enc, body_box["out"], f0, s0)
                    body_box["masked"] = lp0
                    body_box["guess"] = top_moves(enc["legal"], lp0, 3)
                    start_spec(body_box["guess"])
                    maybe_spec3()
                body_box["tc"] = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                body_box["err"] = exc
        bt = threading.Thread(target=_body, daemon=True)
        sw = sys.getswitchinterval()
        sys.setswitchinterval(SEARCH_SWITCH_INTERVAL)   # the body thread (GPU / CPU network work) waits less for the interpreter lock
        bt.start()
        tb0 = time.time()
        base = self.eng.evaluate(board, self.v2_base, on_depth=maybe_spec3 if self.spec3 is not None else None)
        tm["sf_pass1"] = time.time() - tb0
        bt.join()
        sys.setswitchinterval(sw)
        if "err" in body_box:
            raise body_box["err"]
        if self.next_body is not None:   # the next search's body pass runs while this one waits for its complements
            job, self.next_body = self.next_body, None
            try:
                self.prefetch_body(*job)
            except Exception:  # noqa: BLE001
                self._prebody = None
        if body_box.get("prefetched"):
            tm["body_prefetched"] = 1.0
        enc = body_box["enc"]
        tm["encode"] = body_box.get("t_encode")
        tm["body"] = body_box["t"]
        tm["candidates_known"] = body_box.get("tc")
        if base is None:   # stopped
            with s3["lock"]:
                if s3["th"] is not None:
                    s3["cancel"].set()
            return None
        if source == "Network" and self.cand_pass == "masked":
            guess = body_box["guess"]
            res["masked_logp"] = body_box["masked"]
        elif source == "Network":   # base150k: B's heads on the pass-1 values of every legal move
            tc = time.time()
            fb, sb = self.net.features(enc, base, E)
            lpb, _ = self.net.heads(enc, body_box["out"], fb, sb)
            guess = top_moves(enc["legal"], lpb, 3)
            res["masked_logp"] = lpb
            tm["candidates_pass"] = time.time() - tc
        res["path"]["candidate_pass"] = self.cand_pass if source == "Network" else "-"
        cands = v2_candidates(guess, base)
        if on_base is not None:
            def _lf_get():
                if lth is None:
                    return None
                lth.join(2.0)
                return None if (lth.is_alive() or "err" in lbox) else lbox.get("lf")
            try:
                on_base(enc, body_box["out"], dict(base), _lf_get)
            except Exception:  # noqa: BLE001
                pass
        ext = None
        t2 = time.time()
        if len(cands) >= 2:
            if th is not None and cands == guess:
                th.join()
                ext = sres.get("v")
                tm["sf_pass2_spec"] = sres.get("t")
            tm["spec_used"] = 1.0 if ext is not None else 0.0
            if ext is None and self.spec3 is not None:
                with s3["lock"]:
                    hit = s3["th"] is not None and s3["target"] == cands
                if hit:
                    s3["th"].join()
                    ext = s3["box"].get("v")
                    tm["sf_pass2_spec3"] = s3["box"].get("t")
                tm["spec3_used"] = 1.0 if (hit and ext is not None) else 0.0
                tm["spec3_starts"] = float(s3["starts"])
            if ext is None:
                ext = self.eng.extend(board, cands, self.v2_ext)
            if ext is None:
                return None
        tm["sf_pass2_wait"] = time.time() - t2
        with s3["lock"]:   # an obsolete speculation is stopped now, so the third engine is free for the next move
            if s3["th"] is not None and s3["th"].is_alive():
                s3["cancel"].set()
                s3["th"].join()
        v = dict(base)
        if ext:
            v.update({u: cp for u, cp in ext.items() if u in v})
        missing = [u for u in enc["legal"] if u not in v]
        if missing:
            raise RuntimeError(f"Stockfish gave no value for {missing[:3]}")
        lf = None
        if lth is not None:
            tl0 = time.time()
            lth.join()
            tm["leela_wait"] = time.time() - tl0
            if "err" in lbox:
                res["path"].update(head="off", leela=f"failed: {lbox['err']}")
            else:
                lf = lbox["lf"]
                tm["leela"] = lbox["t"]
                res["path"].update(head="on", leela=f"lc0 {leela.nodes} nodes {leela.backend}, {lbox['info']['visited']} moves visited" + (f", {leela.gpu}" if leela.gpu else ""))
        if before_heads is not None:
            tw = time.time()
            try:
                before_heads()
            except Exception:  # noqa: BLE001
                pass
            tm["children_wait"] = time.time() - tw
        th2 = time.time()
        f, s = self.net.features(enc, v, E)
        res["logp"], res["wdl"] = self.net.heads(enc, body_box["out"], f, s, lf)
        res["lfeat"] = lf
        tm["heads"] = time.time() - th2
        tm["total"] = time.time() - t0
        res.update(values=v, base=base, candidates=cands, timings=tm,
                   v2info=(max(base, key=base.get), max(v, key=v.get), {u: (base[u], cp) for u, cp in (ext or {}).items() if u in base}))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "export-head", "onnx", "info"])
    ap.add_argument("src")
    ap.add_argument("--onnx")
    ap.add_argument("--pb")
    ap.add_argument("--out")
    ap.add_argument("--option", default="C", choices=["C", "Cmd", "none"])
    ap.add_argument("--no-vpath", action="store_true")
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--run-meta")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--base-model", help="export-head: B's training weights (the head's frozen body)")
    ap.add_argument("--base", help="export-head: B's engine file (toby2_b.pt) exported from --base-model")
    ap.add_argument("--backend", default=LEELA_BACKEND, help="export-head: lc0 backend of the head's training inputs")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.src, a.onnx, a.pb, a.out, a.option, not a.no_vpath, a.freeze, a.run_meta)
    elif a.cmd == "export-head":
        export_head(a.src, a.base_model, a.base, a.out, a.backend, a.run_meta)
    elif a.cmd == "onnx":
        onnx_export(a.src, a.out, a.int8)
    else:
        import torch
        print(json.dumps(torch.load(a.src, map_location="cpu", weights_only=False)["meta"], indent=1))


if __name__ == "__main__":
    main()
