# -*- coding: utf-8 -*-
"""Linux x86-64 build of Toby (CPU PyTorch), from public_release/ (run make_public_release.py first). Windows: build_package.py.
  python build_package_linux.py freeze      PyInstaller: package_build_linux/dist/toby/toby (Python + PyTorch CPU + Maia-3 code + the corrector)
  python build_package_linux.py assemble    package_build_linux/stage/: Toby/ (core), maia/, stockfish/
  python build_package_linux.py archive     package_build_linux/archives/*.tar.gz (core, maia, stockfish) + sha256 and sizes in manifest_part.json
  python build_package_linux.py freeze cuda the same with a CUDA build of PyTorch, into package_build_linux/dist_cuda/toby (Toby-full: Maia on the GPU)
  python build_package_linux.py zip         package_build_linux/zips/Toby-linux-light.zip (+ Toby-linux-full.zip when dist_cuda/ and the Leela
                                            folder exist): one top folder Toby/, executable bits kept; sizes and sha256 in zips/manifest_linux.json
Leela for Toby-full: TOBY_LC0_DIR_LINUX = a folder with an lc0 built with CUDA (lc0 + the CUDA libraries it needs), TOBY_NET = the BT3 network.
Paths (environment): TOBY_MAIA_REPO (clone of github.com/CSSLab/maia3), TOBY_MAIA_WEIGHTS (maia3-79m.pt), TOBY_STOCKFISH (Stockfish 19 Linux binary).
Leela: Toby-linux-full uses TOBY_LC0_DIR_LINUX (lc0 built with CUDA); Toby-linux-light uses TOBY_LC0_CPU_DIR_LINUX (a CPU build) with TOBY_LC0_CPU_BACKEND, both with TOBY_NET."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUB = ROOT / "public_release"
OUT = ROOT / "package_build_linux"
MAIA_REPO = Path(os.environ.get("TOBY_MAIA_REPO", "/opt/toby-build/maia3"))
MAIA_WEIGHTS = Path(os.environ.get("TOBY_MAIA_WEIGHTS") or next(Path.home().glob(".cache/huggingface/hub/models--UofTCSSLab--Maia3-79M/snapshots/*/maia3-79m.pt"), "/nonexistent/maia3-79m.pt"))
MAIA_FT = Path(os.environ.get("TOBY_MAIA_FT", str(ROOT / "maia" / "maia3-79m-ft-classical.pt")))   # fine-tuned Maia (Classical; Git LFS)
SHIP_MAIA = os.environ.get("TOBY_SHIP_MAIA") == "1"   # 2.2.1: Toby 1.0 / Maia are left out (default); TOBY_SHIP_MAIA=1 = the 2.2 layout with maia/
STOCKFISH = Path(os.environ.get("TOBY_STOCKFISH", "/opt/toby-build/sf19/stockfish/stockfish-linux-x86-64-universal"))
TOBY2_WEIGHTS = Path(os.environ.get("TOBY2_WEIGHTS", str(ROOT / "nets" / "toby2_b.pt")))   # Toby 2's network B (toby2.py export), shipped as nets/toby2_b.pt
# shipped next to it in nets/ (same names in the folder of TOBY2_WEIGHTS): the Standard product file, the level head and its calibration
TOBY2_LIGHT_FILES = ("toby2_standard.json", "toby2_level.pt", "toby2_level_cal.json")
TOBY2_FULL_FILES = ("toby2_accurate.json", "toby2_lhead.pt")   # Toby-linux-full only: the Accurate product file and the Leela head (as Windows full)
# The CUDA libraries the shipped Linux lc0 links (DT_NEEDED; its RUNPATH is /usr/local/cuda/lib64, absent on most machines, and the binary is
# pinned by sha256 in toby2_accurate.json, so it is never patched): kept as real files at the top of _internal/, which the frozen toby's
# bootloader puts on LD_LIBRARY_PATH and lc0 inherits. libcublasLt is libcublas's own dependency.
LC0_CUDA_LIBS = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")


def freeze(variant="cpu"):
    OUT.mkdir(exist_ok=True)
    if variant == "cuda":
        import torch
        if not torch.version.cuda:
            raise SystemExit(f"freeze cuda needs a CUDA build of PyTorch (this one: {torch.__version__})")
    sub, work = ("dist_cuda", "work_cuda") if variant == "cuda" else ("dist", "work")
    slim = ["--exclude-module", "triton"] if (variant == "cuda" and os.environ.get("TOBY_SLIM", "1") == "1") else []   # torch.compile only
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "toby",
           "--distpath", str(OUT / sub), "--workpath", str(OUT / work), "--specpath", str(OUT)] + slim + [
           "--paths", str(MAIA_REPO), "--paths", str(PUB), "--paths", str(PUB / "toby2net"),
           "--hidden-import", "toby2", "--hidden-import", "leela_torch", "--hidden-import", "model", "--hidden-import", "encode", "--hidden-import", "leela_feats", "--hidden-import", "levelgen", "--hidden-import", "levelhead",
           "--hidden-import", "maia3", "--hidden-import", "maia3.dataset", "--hidden-import", "maia3.uci", "--hidden-import", "maia3.models",
           "--hidden-import", "maia3.model_registry", "--hidden-import", "maia3.utils", "--hidden-import", "maia3.presets", "--hidden-import", "corrector", "--hidden-import", "wdl_endgame_p1",
           "--exclude-module", "matplotlib", "--exclude-module", "tkinter", "--exclude-module", "IPython", "--exclude-module", "pytest",
           str(PUB / "toby_uci.py")]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if variant == "cuda" and os.environ.get("TOBY_SLIM", "1") == "1":
        slim_cuda()


def slim_cuda():
    """The Linux CUDA build: PyInstaller copies every linked NVIDIA library to the top of _internal/ while torch's own copies stay under
    _internal/nvidia/*/lib (and triton/), where torch's RPATHs point. Remove the top-level copies that are byte-identical to a nested one
    (about 2.6 GB). Toby must be tested after this (likelihood against the reference, Accurate, Hybrid, on a GPU)."""
    internal = OUT / "dist_cuda" / "toby" / "_internal"
    nested = {}
    for f in internal.rglob("*.so*"):
        if f.parent != internal and f.is_file() and not f.is_symlink():
            nested.setdefault(f.name, []).append(f)
    freed = 0
    for name in LC0_CUDA_LIBS:   # lc0's libraries: a real top-level file (a zip cannot store a link), found by lc0 through LD_LIBRARY_PATH
        f = internal / name
        if f.is_symlink():
            target = f.resolve()
            f.unlink()
            shutil.copy2(target, f)
            print("slim: kept for lc0 (link replaced by a copy)", name, flush=True)
        elif not f.exists():
            src = nested.get(name)
            if not src:
                raise SystemExit(f"slim: {name} (needed by lc0) not found in _internal")
            shutil.copy2(src[0], f)
            print("slim: kept for lc0 (copied from", src[0].relative_to(internal), ")", flush=True)
    for f in sorted(internal.glob("*.so*")):   # PyInstaller's top-level symlinks into _internal (a zip would store each as a full copy)
        if f.name in LC0_CUDA_LIBS:
            continue
        if f.is_symlink():
            target = f.resolve()
            if internal in target.parents and target.is_file():
                freed += target.stat().st_size
                print("slim: removed top-level link", f.name, "->", target.relative_to(internal), flush=True)
                f.unlink()
    for f in sorted(internal.glob("*.so*")):
        if not f.is_file() or f.is_symlink() or f.name not in nested or f.name in LC0_CUDA_LIBS:
            continue
        size = f.stat().st_size
        same = [g for g in nested[f.name] if g.stat().st_size == size and _sha(g) == _sha(f)]
        if same:
            freed += size
            print("slim: removed duplicate", f.name, f"{size / 1e6:.0f} MB (kept {same[0].relative_to(internal)})", flush=True)
            f.unlink()
    print(f"slim: {freed / 1e9:.2f} GB of duplicates removed", flush=True)


def _sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def assemble():
    st = OUT / "stage"
    if st.exists():
        shutil.rmtree(st)
    core = st / "Toby"
    if (OUT / "dist" / "toby").exists():   # the CPU build's core (archive); a full-only build has none (zip_all reads dist_cuda/ itself)
        shutil.copytree(OUT / "dist" / "toby", core)
        shutil.copytree(PUB / "nn", core / "nn")
        for f in ("README.md", "LICENSE", "NOTICE.md"):
            shutil.copy(PUB / f, core / f)
        shutil.copytree(PUB / "docs", core / "docs")
    else:
        st.mkdir(parents=True)
    (st / "maia").mkdir()   # empty without SHIP_MAIA (never zipped then)
    if SHIP_MAIA:
        shutil.copy(MAIA_WEIGHTS, st / "maia" / "maia3-79m.pt")
        if MAIA_FT.exists():
            shutil.copy(MAIA_FT, st / "maia" / "maia3-79m-ft-classical.pt")
    (st / "stockfish").mkdir()
    shutil.copy(STOCKFISH, st / "stockfish" / "stockfish")
    os.chmod(st / "stockfish" / "stockfish", 0o755)
    if not TOBY2_WEIGHTS.exists() or TOBY2_WEIGHTS.stat().st_size < 1_000_000:   # never a Toby 2 package without its network
        raise SystemExit(f"Toby 2 weights missing: {TOBY2_WEIGHTS} (python toby2.py export ...)")
    (st / "toby2").mkdir()
    shutil.copy(TOBY2_WEIGHTS, st / "toby2" / "toby2_b.pt")
    for f in TOBY2_LIGHT_FILES:   # product file (Standard) and the level head + its calibration ('toby level'), next to toby2_b.pt
        src = TOBY2_WEIGHTS.parent / f
        if not src.exists():
            raise SystemExit(f"Toby 2 file missing: {src}")
        shutil.copy(src, st / "toby2" / f)
    full = [TOBY2_WEIGHTS.parent / f for f in TOBY2_FULL_FILES]
    if all(f.exists() for f in full):   # Toby-linux-full only (zip_all adds them to the full zip)
        (st / "toby2_full").mkdir()
        for f in full:
            shutil.copy(f, st / "toby2_full" / f.name)
    print("staged:", st)


def archive():
    st, z = OUT / "stage", OUT / "archives"
    z.mkdir(exist_ok=True)
    parts = {}
    for name, src, arc in (("core", st / "Toby", "Toby"), ("maia", st / "maia", "Toby/maia"), ("stockfish", st / "stockfish", "Toby/stockfish"),
                           ("toby2", st / "toby2", "Toby/nets")):
        path = z / f"Toby-linux-x64-{name}.tar.gz"
        with tarfile.open(path, "w:gz") as tf:
            tf.add(src, arcname=arc)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        parts[name] = {"file": path.name, "size": path.stat().st_size, "sha256": h}
        print(name, f"{path.stat().st_size / 1e6:.0f} MB", h, flush=True)
    (z / "manifest_part.json").write_text(json.dumps({"linux-x64": parts}, indent=1))


LC0_DIR = Path(os.environ.get("TOBY_LC0_DIR_LINUX", "/nonexistent"))
NET = Path(os.environ.get("TOBY_NET", "/opt/toby-build/nets/BT3-768x15x24h-swa-2790000.pb.gz"))
LC0_CPU_DIR = Path(os.environ["TOBY_LC0_CPU_DIR_LINUX"]) if os.environ.get("TOBY_LC0_CPU_DIR_LINUX") else None   # Toby-linux-light's Leela (CPU)
LC0_CPU_BACKEND = os.environ.get("TOBY_LC0_CPU_BACKEND", "")      # the backend that passed the policy gate (e.g. onnx-cpu)
LC0_GPU_BACKEND = os.environ.get("TOBY_LC0_GPU_BACKEND") or "cuda"   # Toby-linux-full: always write toby_backend.txt (cuda by default), so Toby skips the lc0 build at once when no NVIDIA GPU answers
EXEC = ("toby", "stockfish/stockfish", "lc0/lc0")   # paths under Toby/ that must stay executable


def _zip_tree(zf, src, arc_root):
    import zipfile
    for f in sorted(src.rglob("*")):
        if f.is_file():
            arc = str(Path(arc_root) / f.relative_to(src))
            zi = zipfile.ZipInfo.from_file(f, arc)
            zi.compress_type = zipfile.ZIP_DEFLATED
            mode = f.stat().st_mode & 0o777
            if any(arc == f"Toby/{e}" for e in EXEC) or (mode & 0o111):
                mode |= 0o755
            zi.external_attr = (0o100000 | mode) << 16   # regular file + Unix permissions (kept by unzip and by Siby)
            with open(f, "rb") as fh, zf.open(zi, "w", force_zip64=True) as out:
                shutil.copyfileobj(fh, out, 1 << 20)


def toby_version():
    """VERSION of the engine being packaged (public_release/toby_uci.py), written to Toby/VERSION and to the manifest."""
    import re
    return re.search(r'^VERSION = "([^"]+)"', (PUB / "toby_uci.py").read_text(encoding="utf-8"), re.M).group(1)


RELEASE_DATE = os.environ.get("TOBY_RELEASE_DATE") or __import__("time").strftime("%Y%m%d")


def zip_all():
    import zipfile
    st, z = OUT / "stage", OUT / "zips"
    z.mkdir(exist_ok=True)
    if LC0_CPU_DIR is not None and not LC0_CPU_BACKEND:
        raise SystemExit("TOBY_LC0_CPU_BACKEND must name the backend that passed the policy gate")
    variants = {}
    if (OUT / "dist" / "toby").exists():   # a full-only build (freeze cuda alone) makes no light zip
        variants["light"] = (OUT / "dist" / "toby", ["lc0cpu"] if (LC0_CPU_DIR is not None and NET.exists()) else [])
    if (OUT / "dist_cuda" / "toby").exists() and (LC0_DIR / "lc0").exists() and NET.exists():
        variants["full"] = (OUT / "dist_cuda" / "toby", ["lc0", "nets"])
    man = {}
    for name, (eng, extra) in variants.items():
        path = z / f"Toby-linux-{name}-{RELEASE_DATE}.zip"   # dated names: a CDN never serves an older zip under the same name (lesson I9)
        path.unlink(missing_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
            _zip_tree(zf, eng, "Toby")
            for sub in ("nn", "docs"):
                _zip_tree(zf, PUB / sub, f"Toby/{sub}")
            for f in ("README.md", "LICENSE", "NOTICE.md"):
                _zip_tree_file(zf, PUB / f, f"Toby/{f}")
            if SHIP_MAIA:
                _zip_tree(zf, st / "maia", "Toby/maia")
            _zip_tree(zf, st / "stockfish", "Toby/stockfish")
            for f in sorted((st / "toby2").iterdir()):   # toby2_b.pt + TOBY2_LIGHT_FILES
                _zip_tree_file(zf, f, f"Toby/nets/{f.name}")
            if name == "full":   # + TOBY2_FULL_FILES (the Accurate product needs both, and lc0 + BT3 below)
                missing = [f for f in TOBY2_FULL_FILES if not (st / "toby2_full" / f).exists()]
                if missing:
                    raise SystemExit(f"Toby-linux-full: missing {missing} next to {TOBY2_WEIGHTS}")
                for f in TOBY2_FULL_FILES:
                    _zip_tree_file(zf, st / "toby2_full" / f, f"Toby/nets/{f}")
            zv = zipfile.ZipInfo("Toby/VERSION")
            zv.external_attr = (0o100000 | 0o644) << 16
            zf.writestr(zv, toby_version() + "\n")   # Siby reads it (Toby updates from Siby)
            if "lc0" in extra or "lc0cpu" in extra:
                src, bk = (LC0_DIR, LC0_GPU_BACKEND) if "lc0" in extra else (LC0_CPU_DIR, LC0_CPU_BACKEND)
                _zip_tree(zf, src, "Toby/lc0")
                if bk:   # the backend Toby passes to this lc0 (toby_uci.lc0_backend_file)
                    zi = zipfile.ZipInfo("Toby/lc0/toby_backend.txt")
                    zi.external_attr = (0o100000 | 0o644) << 16
                    zf.writestr(zi, bk + "\n")
                _zip_tree_file(zf, NET, f"Toby/nets/{NET.name}")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        man[name] = {"fichier": path.name, "sha256": h.hexdigest(), "taille": path.stat().st_size}
        print(name, f"{path.stat().st_size / 1e6:.0f} MB", h.hexdigest(), flush=True)
    (z / "manifest_linux.json").write_text(json.dumps({"version": toby_version(), "linux": man}, indent=1))   # merged by make_toby_manifest.py


def _zip_tree_file(zf, f, arc):
    import zipfile
    zi = zipfile.ZipInfo.from_file(f, arc)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (0o100000 | 0o644) << 16
    with open(f, "rb") as fh, zf.open(zi, "w", force_zip64=True) as out:
        shutil.copyfileobj(fh, out, 1 << 20)


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2] if len(sys.argv) > 2 else "cpu")
    else:
        {"assemble": assemble, "archive": archive, "zip": zip_all}[sys.argv[1]]()
