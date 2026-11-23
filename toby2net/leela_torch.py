# -*- coding: utf-8 -*-
"""Body B of the v7 pilot (nn/PROTOCOL_V7.md amendment 1): an official Leela Chess Zero BT4 network in PyTorch.

The network file (lczero.org .pb.gz) is exported by lc0's own converter (`lc0 leela2onnx`); this module rebuilds the same computation
in PyTorch from the ONNX graph's weights (our code, written from lc0's GPL v3 converter `src/neural/onnx/converter.cc` and encoder
`src/neural/encoder.cc`: architecture and input encoding are lc0's, credited in NOTICE.md), and is checked against onnxruntime on the
exported graph and against the lc0 binary (`python leela_torch.py check`).
  pb_info(path)            -> licence string and network format of a .pb.gz (input format, activations...) without lc0
  lc0_planes(boards, fmt)  -> lc0's 112 input planes for a game history (python-chess boards, oldest first), as 109 uint64 masks
                              + rule-50 ply + transform (EncodePositionForNN with FillEmptyHistory::FEN_ONLY, as lc0's search)
  Lc0Net.from_onnx(path)   -> torch module: body(planes) -> (B, 64, E); policy_grid(x) -> (B, 67 * 64) as lc0's attention policy
                              before its gather; value(x) -> (B, 3) logits (W, D, L)
  vocab_index(transform)   -> for each move of the common Maia-3 / v7 vocabulary (mover's frame), its index in the policy grid"""
import gzip
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------ .pb.gz header (no lc0 needed)


def _varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return x, i


def _fields(b):
    i, out = 0, []
    while i < len(b):
        key, i = _varint(b, i)
        f, t = key >> 3, key & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif t == 5:
            v = b[i:i + 4]
            i += 4
        elif t == 1:
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"wire type {t}")
        out.append((f, t, v))
    return out


NF_NAMES = {1: "input", 2: "output", 3: "network", 4: "policy", 5: "value", 6: "moves_left", 7: "default_activation",
            8: "smolgen_activation", 9: "ffn_activation", 10: "input_embedding"}


def pb_info(path):
    """{'license': str, 'input': int, ...} from the Net message (field 2 licence, field 4 format -> 2 network_format)."""
    with gzip.open(path, "rb") as fh:
        head = fh.read(4096)   # magic, licence, min_version and format come first in lc0's files
    out = {}
    for f, t, v in _fields_prefix(head):
        if f == 2 and t == 2:
            out["license"] = bytes(v).decode("utf-8", "replace")
        if f == 4 and t == 2:
            for f2, t2, v2 in _fields(v):
                if f2 == 2 and t2 == 2:
                    for f3, t3, v3 in _fields(v2):
                        out[NF_NAMES.get(f3, f3)] = v3
                if f2 == 1:
                    out["weights_encoding"] = v2
    return out


def _fields_prefix(b):
    """Top-level fields of a truncated message: stops at the first field that runs past the buffer (the weights)."""
    i, out = 0, []
    while i < len(b):
        try:
            key, j = _varint(b, i)
            f, t = key >> 3, key & 7
            if t == 0:
                v, j = _varint(b, j)
            elif t == 2:
                n, j = _varint(b, j)
                if j + n > len(b):
                    break
                v = b[j:j + n]
                j += n
            elif t == 5:
                v = b[j:j + 4]
                j += 4
            elif t == 1:
                v = b[j:j + 8]
                j += 8
            else:
                break
        except IndexError:
            break
        out.append((f, t, v))
        i = j
    return out


# ------------------------------------------------------------------------------------------------ lc0 input encoding

CLASSICAL, CASTLING_PLANE, CANON, CANON_HECTO, CANON_HECTO_ARM, CANON_V2, CANON_V2_ARM = 1, 2, 3, 4, 132, 5, 133
FLIP, MIRROR, TRANSPOSE = 1, 2, 4
ALL = (1 << 64) - 1
AUX = 104


def rev_bits_in_bytes(v):
    v = ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1)
    v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0F) | ((v & 0x0F0F0F0F0F0F0F0F) << 4)
    return v & ALL


def rev_bytes(v):
    return int.from_bytes((v & ALL).to_bytes(8, "little")[::-1], "little")


def transpose_bits(v):
    v = (v & 0xAA00AA00AA00AA00) >> 9 | (v & 0x0055005500550055) << 9 | (v & 0x55AA55AA55AA55AA)
    v = (v & 0xCCCC0000CCCC0000) >> 18 | (v & 0x0000333300003333) << 18 | (v & 0x3333CCCC3333CCCC)
    v = (v & 0xF0F0F0F000000000) >> 36 | (v & 0x000000000F0F0F0F) << 36 | (v & 0x0F0F0F0FF0F0F0F0)
    return v & ALL


def is_canonical(fmt):
    return fmt >= CANON


class Pos:
    """One position of the history, in lc0's terms: masks in the frame of its own side to move (ranks mirrored for Black)."""
    __slots__ = ("ours", "theirs", "pawns", "knights", "bishops", "rooks", "queens", "kings", "castle", "ep", "r50", "black", "reps", "key")


def lc0_history(moves, board=None):
    """Pos list for the start position and after each move (python-chess), with lc0's en passant flag (set only when a pawn of the
    side to move attacks the passed square, board.cc) and repetitions (PositionHistory::ComputeLastMoveRepetitions).
    board: the starting position (a python-chess board, e.g. a GUI's 'position fen'; default the standard start, as the training data)."""
    import chess
    b = chess.Board() if board is None else chess.Board(board.fen())
    out = []

    def make(board, ep_file):
        p = Pos()
        black = board.turn == chess.BLACK
        f = rev_bytes if black else (lambda v: v)
        us, them = board.turn, not board.turn
        p.ours, p.theirs = f(int(board.occupied_co[us])), f(int(board.occupied_co[them]))
        p.pawns, p.knights, p.bishops = f(int(board.pawns)), f(int(board.knights)), f(int(board.bishops))
        p.rooks, p.queens, p.kings = f(int(board.rooks)), f(int(board.queens)), f(int(board.kings))
        c = board.castling_rights
        wq, wk, bq, bk = bool(c & chess.BB_A1), bool(c & chess.BB_H1), bool(c & chess.BB_A8), bool(c & chess.BB_H8)
        p.castle = (bq, bk, wq, wk) if black else (wq, wk, bq, bk)   # (we 000, we 00, they 000, they 00)
        p.ep = ep_file
        p.r50 = board.halfmove_clock
        p.black = black
        p.key = (p.ours, p.theirs, p.pawns, p.knights, p.bishops, p.rooks, p.queens, p.kings, p.castle, p.ep, black)
        return p
    ep0 = None
    if b.ep_square is not None and b.has_legal_en_passant():   # a FEN start with a capturable en passant square (never the standard start)
        ep0 = chess.square_file(b.ep_square)
    out.append(make(b, ep0))
    out[-1].reps = 0
    for u in moves:
        m = chess.Move.from_uci(u)
        double = b.piece_type_at(m.from_square) == chess.PAWN and abs(m.to_square - m.from_square) == 16
        b.push(m)
        ep = None
        if double:
            sq = (m.from_square + m.to_square) // 2
            if b.attackers_mask(b.turn, sq) & b.pawns & b.occupied_co[b.turn]:
                ep = chess.square_file(m.to_square)
        p = make(b, ep)
        reps = 0
        if p.r50 >= 4:   # ComputeLastMoveRepetitions
            for k in range(len(out) - 4, -1, -2):
                q = out[k]
                if q.key == p.key:
                    reps = 1 + q.reps
                    break
                if q.r50 < 2:
                    break
        p.reps = reps
        out.append(p)
    return out


def choose_transform(p):
    if any(p.castle):
        return 0
    our_king = p.kings & p.ours
    t = 0
    if our_king & 0x0F0F0F0F0F0F0F0F:
        t |= FLIP
        our_king = rev_bits_in_bytes(our_king)
    if p.pawns:
        return t
    if our_king & 0xFFFFFFFF00000000:
        t |= MIRROR
        our_king = rev_bytes(our_king)
    if our_king & 0xE0C08000:
        t |= TRANSPOSE
    elif our_king & 0x10204080:
        for bb in (p.ours | p.theirs, p.ours, p.kings, p.queens, p.rooks, p.knights, p.bishops):
            v = bb
            if t & FLIP:
                v = rev_bits_in_bytes(v)
            if t & MIRROR:
                v = rev_bytes(v)
            alt = transpose_bits(v)
            if v < alt:
                return t
            if v > alt:
                return t | TRANSPOSE
    return t


def apply_transform(v, t):
    if t & FLIP:
        v = rev_bits_in_bytes(v)
    if t & MIRROR:
        v = rev_bytes(v)
    if t & TRANSPOSE:
        v = transpose_bits(v)
    return v


def lc0_planes(hist, fmt, n_hist=8):
    """hist: Pos list up to the current position (lc0_history(moves)[:ply + 1]). -> (masks (109,) uint64 = planes 0..108, rule-50 ply,
    transform). Planes 109-111 are built at batch time: rule 50 (ply / 100 for hectoplies formats, else the ply), armageddon side
    (all set when Black is to move, armageddon formats), ones."""
    m = [0] * (AUX + 5)
    cur = hist[-1]
    canon = is_canonical(fmt)
    t = choose_transform(cur) if canon else 0
    if fmt == CLASSICAL:
        for k in range(4):
            if cur.castle[k]:
                m[AUX + k] = ALL
    else:
        we000, we00, th000, th00 = cur.castle
        m[AUX + 0] = (1 if we000 else 0) | ((1 << 56) if th000 else 0)
        m[AUX + 1] = ((1 << 7) if we00 else 0) | ((1 << 63) if th00 else 0)
    if canon:
        m[AUX + 4] = (1 << (56 + cur.ep)) if cur.ep is not None else 0
    elif cur.black:
        m[AUX + 4] = ALL
    skip_non_repeats = fmt in (CANON_V2, CANON_V2_ARM)
    n = len(hist) - 1
    i, idx = 0, n
    while i < min(n_hist, 8):
        if idx < 0:
            break   # FillEmptyHistory::FEN_ONLY from the start position
        p = hist[idx]
        same_side = (p.black == cur.black)
        if canon and p.castle != (cur.castle if same_side else (cur.castle[2], cur.castle[3], cur.castle[0], cur.castle[1])):
            break
        if canon and idx != n and p.ep is not None:
            break
        if skip_non_repeats and p.reps == 0 and i > 0:
            if p.r50 == 0:
                break
            idx -= 1
            continue
        mir = (lambda v: v) if same_side else rev_bytes   # lc0 mirrors the history boards into the current side's frame
        ours, theirs = (p.ours, p.theirs) if same_side else (p.theirs, p.ours)
        ours, theirs = mir(ours), mir(theirs)
        pcs = [mir(p.pawns), mir(p.knights), mir(p.bishops), mir(p.rooks), mir(p.queens), mir(p.kings)]
        base = i * 13
        for k in range(6):
            m[base + k] = ours & pcs[k]
            m[base + 6 + k] = theirs & pcs[k]
        if p.reps >= 1:
            m[base + 12] = ALL
        i += 1
        idx -= 1
        if canon and p.r50 == 0:
            break
    if t:
        m = [v if v in (0, ALL) else apply_transform(v, t) for v in m]
    return np.array(m, dtype=np.uint64), cur.r50, t


def planes_tensor(masks, r50, black, fmt):
    """masks (B, 109) int64, r50 (B,), black (B,) -> (B, 112, 8, 8) float in lc0's layout (plane, rank, file)."""
    B = masks.shape[0]
    bits = ((masks.unsqueeze(-1) >> torch.arange(64, device=masks.device)) & 1).float()   # (B, 109, 64)
    hecto = fmt in (CANON_HECTO, CANON_HECTO_ARM, CANON_V2, CANON_V2_ARM)
    r = (r50.float() / 100.0) if hecto else r50.float()
    arm = black.float() if fmt in (CANON_HECTO_ARM, CANON_V2_ARM) else torch.zeros_like(r)
    tail = torch.stack([r, arm, torch.ones_like(r)], 1).unsqueeze(-1).expand(B, 3, 64)
    return torch.cat([bits, tail], 1).reshape(B, 112, 8, 8)


# ------------------------------------------------------------------------------------------------ the network


def _act(name):
    return {"mish": F.mish, "swish": F.silu, "relu": F.relu, "selu": F.selu, "none": (lambda x: x),
            "sqrrelu": (lambda x: F.relu(x) ** 2)}[name]


class LN(nn.Module):
    def __init__(self, g, b, eps):
        super().__init__()
        self.ln = nn.LayerNorm(len(g), eps=eps)
        with torch.no_grad():
            self.ln.weight.copy_(torch.as_tensor(g))
            self.ln.bias.copy_(torch.as_tensor(b))

    def forward(self, x):
        return self.ln(x)


def _lin(w, b=None):
    """lc0 MatMul weights (in, out) -> nn.Linear."""
    w = torch.as_tensor(np.ascontiguousarray(w), dtype=torch.float32)
    lin = nn.Linear(w.shape[0], w.shape[1], bias=b is not None)
    with torch.no_grad():
        lin.weight.copy_(w.t())
        if b is not None:
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32).reshape(-1))
    return lin


class Encoder(nn.Module):
    def __init__(self, W, name, heads, eps, alpha, act, ffn_act, smol_act, smol_w):
        super().__init__()
        g = lambda k: W.get(f"{name}/{k}")
        self.q, self.k, self.v = _lin(g("mha/Q/w"), g("mha/Q/b")), _lin(g("mha/K/w"), g("mha/K/b")), _lin(g("mha/V/w"), g("mha/V/b"))
        self.dense = _lin(g("mha/out/dense/w"), g("mha/out/dense/b"))
        self.heads, self.alpha = heads, alpha
        self.has_smol = g("smolgen/compress") is not None
        if self.has_smol:
            self.compress = _lin(g("smolgen/compress"))
            self.s_d1 = _lin(g("smolgen/dense1/w"), g("smolgen/dense1/b"))
            self.s_ln1 = LN(*W.ln(f"{name}/smolgen/ln1"), 1e-3)
            self.s_d2 = _lin(g("smolgen/dense2/w"), g("smolgen/dense2/b"))
            self.s_ln2 = LN(*W.ln(f"{name}/smolgen/ln2"), 1e-3)
            self.smol_w = smol_w   # shared (gen, 4096) parameter holder
        self.ln1 = LN(*W.ln(f"{name}/ln1"), eps)
        self.f1, self.f2 = _lin(g("ffn/dense1/w"), g("ffn/dense1/b")), _lin(g("ffn/dense2/w"), g("ffn/dense2/b"))
        self.ln2 = LN(*W.ln(f"{name}/ln2"), eps)
        self.act, self.ffn_act, self.smol_act = act, ffn_act, smol_act

    def forward(self, x):
        B = x.shape[0]
        H = self.heads
        d = self.q.out_features // H
        q = self.q(x).view(B, 64, H, d).transpose(1, 2)
        k = self.k(x).view(B, 64, H, d).transpose(1, 2)
        v = self.v(x).view(B, 64, H, d).transpose(1, 2)
        s = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(d))
        if self.has_smol:
            h = self.compress(x).reshape(B, -1)
            h = self.s_ln1(_act(self.smol_act)(self.s_d1(h)))
            h = self.s_ln2(_act(self.smol_act)(self.s_d2(h))).view(B, H, -1)
            s = s + (h @ self.smol_w.w).view(B, H, 64, 64)
        o = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, 64, H * d)
        x = self.ln1(x + self.dense(o) * self.alpha)
        y = self.f2(_act(self.ffn_act)(self.f1(x))) * self.alpha
        return self.ln2(x + y)


class _Holder(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(np.ascontiguousarray(w), dtype=torch.float32))


class OnnxWeights:
    """Initializers of the lc0 export, addressed by the node name that consumes them."""

    def __init__(self, path):
        import onnx
        from onnx import numpy_helper
        m = onnx.load(path)
        self.init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
        self.nodes = {n.name: n for n in m.graph.node}
        self.by_node = {}
        for n in m.graph.node:
            ins = [self.init[i] for i in n.input if i in self.init]
            if ins:
                self.by_node[n.name] = ins
        self.ops = {n.name: n.op_type for n in m.graph.node}
        self.graph = m

    def get(self, name):
        v = self.by_node.get(name)
        if v is None:
            return None
        return v[0]

    def ln(self, name):
        """(gamma, beta) of a layer norm: LayerNormalization node, or lc0's alternative form (Mul '/gammas', Add '/betas')."""
        if name in self.by_node and self.ops.get(name) == "LayerNormalization":
            v = self.by_node[name]
            return v[0], v[1]
        return self.get(name + "/gammas"), self.get(name + "/betas")

    def act_of(self, prefix):
        for a in ("mish", "swish", "relu", "selu", "sqrrelu"):
            if any(n.startswith(f"{prefix}/{a}") for n in self.nodes):
                return a
        return "none"


class Lc0Net(nn.Module):
    @classmethod
    def from_onnx(cls, path, info=None):
        W = OnnxWeights(path)
        return cls(W, info or {})

    def __init__(self, W, info):
        super().__init__()
        self.fmt = int(info.get("input", CANON_HECTO_ARM))
        enc_names = sorted({n.split("/")[1] for n in W.nodes if n.startswith("/encoder")}, key=lambda s: int(s[7:]))
        self.n_enc = len(enc_names)
        pre_w = W.get("/attn_body/embedding/preprocess/matmul")
        self.dense = pre_w is not None      # INPUT_EMBEDDING_PE_DENSE (else PE_MAP: a fixed 64-channel position encoding)
        if self.dense:
            self.pre = _lin(pre_w, W.get("/attn_body/embedding/preprocess/add"))
            self.dense_sz = pre_w.shape[1] // 64
        else:
            pe = W.init["/const/pos_encoding"].reshape(64, 64)
            self.register_buffer("pos_enc", torch.as_tensor(np.ascontiguousarray(pe), dtype=torch.float32), persistent=False)
        self.emb = _lin(W.get("/attn_body/matmul"), W.get("/attn_body/add"))
        E = self.emb.out_features
        self.E = E
        self.act = W.act_of("/attn_body")
        self.alpha = (2.0 * self.n_enc) ** -0.25
        g = W.get("/ip_mul_gate")
        self.mult_gate = nn.Parameter(torch.as_tensor(g).reshape(64, E).float()) if g is not None else None
        g = W.get("/ip_add_gate")
        self.add_gate = nn.Parameter(torch.as_tensor(g).reshape(64, E).float()) if g is not None else None
        if self.dense:
            self.emb_ln = LN(*W.ln("/attn_body/ln"), 1e-3)
            self.ef1 = _lin(W.get("/attn_body/ffn/dense1/w"), W.get("/attn_body/ffn/dense1/b"))
            self.ef2 = _lin(W.get("/attn_body/ffn/dense2/w"), W.get("/attn_body/ffn/dense2/b"))
            self.emb_ffn_act = W.act_of("/attn_body/ffn/dense1")
            self.emb_ln2 = LN(*W.ln("/attn_body/ln2"), 1e-3)
        self.smol = _Holder(W.init["/const/smolgen_w"]) if "/const/smolgen_w" in W.init else None
        heads = W.get("/const/encoder0/mha/shape") if False else None
        shape = W.init.get("/const/encoder0/mha/shape")
        heads = int(shape[2]) if shape is not None else 32
        eps = 1e-3 if self.dense else 1e-6   # default_eps_ of converter.cc
        e0 = enc_names[0]
        self.enc = nn.ModuleList(Encoder(W, f"/{n}", heads, eps, self.alpha, self.act, W.act_of(f"/{e0}/ffn/dense1"),
                                         W.act_of(f"/{e0}/smolgen/dense1"), self.smol) for n in enc_names)
        # attention policy head
        self.p_d1 = _lin(W.get("/policy/dense1/matmul"), W.get("/policy/dense1/add"))
        self.p_act = W.act_of("/policy/dense1")
        self.p_q = _lin(W.get("/policy/Q/matmul"), W.get("/policy/Q/add"))
        self.p_k = _lin(W.get("/policy/K/matmul"), W.get("/policy/K/add"))
        self.p_pro = _lin(W.get("/policy/promotion/matmul"))
        # value head
        self.v_emb = _lin(W.get("/value/embed/matmul"), W.get("/value/embed/add"))
        self.v_d1 = _lin(W.get("/value/dense1/matmul"), W.get("/value/dense1/add"))
        self.v_d2 = _lin(W.get("/value/dense2/matmul"), W.get("/value/dense2/add"))
        mt = W.init.get("/const/mapping_table")
        self.register_buffer("mapping", torch.as_tensor(mt.astype(np.int64)) if mt is not None else torch.zeros(0, dtype=torch.long),
                             persistent=False)

    def embed(self, planes, extra=None, extra_lin=None):
        """planes (B, 112, 8, 8) -> (B, 64, E) after the embedding block. extra (B, X) enters through extra_lin (zero-initialised)."""
        B = planes.shape[0]
        x = planes.permute(0, 2, 3, 1).reshape(B, 64, 112)
        if self.dense:
            pos = self.pre(x[..., :12].reshape(B, 64 * 12)).reshape(B, 64, self.dense_sz)
        else:
            pos = self.pos_enc.to(x.dtype).unsqueeze(0).expand(B, 64, 64)
        h = self.emb(torch.cat([x, pos], -1))
        if extra is not None:
            h = h + extra_lin(extra).unsqueeze(1)
        h = _act(self.act)(h)
        if self.dense:
            h = self.emb_ln(h)
        if self.mult_gate is not None:
            h = h * self.mult_gate
        if self.add_gate is not None:
            h = h + self.add_gate
        if not self.dense:
            return h
        return self.emb_ln2(h + self.ef2(_act(self.emb_ffn_act)(self.ef1(h))) * self.alpha)

    def body(self, planes, extra=None, extra_lin=None, inject=None, inject_at=None):
        x = self.embed(planes, extra, extra_lin)
        for i, e in enumerate(self.enc):
            if inject is not None and i == inject_at:
                x = x + inject.unsqueeze(1)
            x = e(x)
        return x

    def policy_grid(self, x):
        """(B, 64, E) -> (B, 67 * 64): the 64 x 64 from-to logits then the 3 x 64 promotion rows, as lc0 before its gather."""
        B = x.shape[0]
        h = _act(self.p_act)(self.p_d1(x))
        q, k = self.p_q(h), self.p_k(h)
        d = q.shape[-1]
        grid = q @ k.transpose(1, 2) * (1.0 / math.sqrt(d))   # (B, 64, 64)
        pro = self.p_pro(k[:, 56:64, :]).transpose(1, 2)     # (B, 4, 8)
        pro = (pro[:, :3, :] + pro[:, 3:4, :]).transpose(1, 2).reshape(B, 1, 24)
        sl = grid[:, 48:56, 56:64].reshape(B, 64, 1).expand(B, 64, 3).reshape(B, 8, 24)
        prom = (sl + pro).reshape(B, 3, 64)
        return torch.cat([grid, prom], 1).reshape(B, 67 * 64)

    def value(self, x, add=None):
        B = x.shape[0]
        h = _act(self.act)(self.v_emb(x)).reshape(B, -1)
        h = self.v_d1(h)
        if add is not None:
            h = h + add
        return self.v_d2(_act(self.act)(h))   # logits (W, D, L)


def _sq_t(sq, t):
    r, f = divmod(sq, 8)
    if t & FLIP:
        f = 7 - f
    if t & MIRROR:
        r = 7 - r
    if t & TRANSPOSE:
        r, f = f, r
    return r * 8 + f


def vocab_index(t):
    """(4352,) long: for each move of the v7 vocabulary (mover's frame; a1 = 0; promotions f7 t8 q / r / b / n), its index in the
    policy grid of transform t (from-to entry f * 64 + t; promotions q / r / b in the 3 x 64 rows, knight = the plain entry, as lc0)."""
    out = np.zeros(4352, np.int64)
    for i in range(4096):
        a, b = divmod(i, 64)
        out[i] = _sq_t(a, t) * 64 + _sq_t(b, t)
    j = 4096
    for ff in range(8):
        for tf in range(8):
            for p in range(4):
                fsq, tsq = _sq_t(48 + ff, t), _sq_t(56 + tf, t)
                if p == 3:
                    out[j] = fsq * 64 + tsq
                else:
                    ff2, tf2 = fsq % 8, tsq % 8
                    out[j] = 4096 + ff2 * 24 + tf2 * 3 + p
                j += 1
    return torch.as_tensor(out)


VOCAB_INDEX = None


def vocab_tables():
    global VOCAB_INDEX
    if VOCAB_INDEX is None:
        VOCAB_INDEX = torch.stack([vocab_index(t) for t in range(8)])
    return VOCAB_INDEX


def castle_fix(idx, planes):
    """lc0's policy writes castling as king takes rook (e1h1, e1a1 in the mover's frame); the v7 vocabulary as Maia (e1g1, e1c1). For the
    positions whose own king stands on e1 (planes[:, 5] = our king, step 0; castling rights forbid any canonical transform) the
    vocabulary entries e1g1 / e1c1 (necessarily castling then) read the king-takes-rook entries of the grid. idx (B, NV) grid indices."""
    king_e1 = planes[:, 5, 0, 4] > 0.5
    if king_e1.any():
        idx = idx.clone()
        idx[king_e1, 4 * 64 + 6] = 4 * 64 + 7
        idx[king_e1, 4 * 64 + 2] = 4 * 64 + 0
    return idx
