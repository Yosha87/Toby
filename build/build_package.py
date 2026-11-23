# -*- coding: utf-8 -*-
"""Builds the downloadable Toby packages from public_release/ (run make_public_release.py first).
  python build_package.py freeze     PyInstaller: package_build/dist/toby/toby.exe (Python + PyTorch CPU + the Maia-3 code inside), icon Images/toby.ico
  python build_package.py freeze cuda   the same with a CUDA build of PyTorch (run it from a Python with a CUDA build of PyTorch), into
                                     package_build/dist/toby_cuda/: Toby-full then uses it (Maia on the NVIDIA GPU); Toby-core / Toby-light keep the CPU one
  python build_package.py assemble   package_build/stage/: core + components, the four folders ready to zip
  python build_package.py zip        package_build/zips/*.zip (Toby-core, Toby-Maia, Toby-Leela, Toby-full)
Components (taken from the local installation, see the paths below): Maia-3 weights, lc0 + CUDA libraries, BT3 network."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUB = ROOT / "public_release"
OUT = ROOT / "package_build"
MAIA_REPO = Path(os.environ.get("TOBY_MAIA_REPO_WIN", r"C:\Toby\maia3"))
MAIA_WEIGHTS = Path(os.environ.get("TOBY_MAIA_WEIGHTS_WIN") or r"C:\Toby\.cache\huggingface\hub\models--UofTCSSLab--Maia3-79M\snapshots\a107d6ceb7b298cb04ae1da4edffe2939858b894\maia3-79m.pt")
MAIA_FT = ROOT / "maia" / "maia3-79m-ft-classical.pt"   # fine-tuned Maia (Classical; Git LFS): Toby 1.0 only (and Toby 2 with Candidates = Maia C)
SHIP_MAIA = os.environ.get("TOBY_SHIP_MAIA") == "1"   # 2.2.1: Toby 1.0 / Maia are left out (default); TOBY_SHIP_MAIA=1 = the 2.2 layout with maia/ and Toby-core
STOCKFISH = Path(os.environ.get("TOBY_STOCKFISH_WIN", str(ROOT / "stockfish" / "stockfish.exe")))   # Stockfish 19 for Windows (official build), required by Toby 2
LC0_DIR = Path(os.environ.get("TOBY_LC0_DIR_WIN", r"C:\Toby\lc0"))
# the NVIDIA libraries may be redistributed with the lc0 package as a whole (lc0 README.txt): every file of the official package except the networks and the zip itself
LC0_FILES = [f.name for f in sorted(LC0_DIR.iterdir()) if f.is_file() and not f.name.endswith(".pb.gz") and not f.name.endswith(".zip")]
NET = LC0_DIR / "BT3-768x15x24h-swa-2790000.pb.gz"
# Toby-light's Leela (CPU): a folder with a CPU build of lc0 and its libraries; the backend that passed the policy gate is written next to it
LC0_CPU_DIR = Path(os.environ["TOBY_LC0_CPU_DIR_WIN"]) if os.environ.get("TOBY_LC0_CPU_DIR_WIN") else None
LC0_CPU_BACKEND = os.environ.get("TOBY_LC0_CPU_BACKEND", "")      # e.g. onnx-cpu, dnnl, blas
TOBY2_WEIGHTS = Path(os.environ.get("TOBY2_WEIGHTS_WIN", str(ROOT / "nets" / "toby2_b.pt")))   # Toby 2's network B (toby2.py export) -> nets/toby2_b.pt
LC0_GPU_BACKEND = os.environ.get("TOBY_LC0_GPU_BACKEND") or "cuda"   # Toby-full: always write toby_backend.txt (cuda by default), so Toby knows the lc0 build needs an NVIDIA GPU and skips it at once when none answers


def freeze(variant="cpu"):
    OUT.mkdir(exist_ok=True)
    if variant == "cuda":
        import torch
        if not torch.version.cuda:
            raise SystemExit(f"freeze cuda needs a CUDA build of PyTorch (this one: {torch.__version__})")
        if not torch.cuda.is_available():
            print("note: no visible GPU here; the CUDA build is made anyway, test it on a PC with an NVIDIA GPU before publishing")
    sub = "dist_cuda" if variant == "cuda" else "dist"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "toby", "--icon", str(ROOT / "Images" / "toby.ico"),
           "--distpath", str(OUT / sub), "--workpath", str(OUT / ("work_cuda" if variant == "cuda" else "work")), "--specpath", str(OUT),
           "--paths", str(MAIA_REPO), "--paths", str(PUB), "--paths", str(PUB / "toby2net"),
           "--hidden-import", "toby2", "--hidden-import", "leela_torch", "--hidden-import", "model", "--hidden-import", "encode", "--hidden-import", "leela_feats", "--hidden-import", "levelgen", "--hidden-import", "levelhead",
           "--hidden-import", "maia3", "--hidden-import", "maia3.dataset", "--hidden-import", "maia3.uci", "--hidden-import", "maia3.models",
           "--hidden-import", "maia3.model_registry", "--hidden-import", "maia3.utils", "--hidden-import", "maia3.presets", "--hidden-import", "corrector", "--hidden-import", "wdl_endgame_p1",
           "--exclude-module", "matplotlib", "--exclude-module", "tkinter", "--exclude-module", "IPython", "--exclude-module", "pytest",
           str(PUB / "toby_uci.py")]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def assemble():
    st = OUT / "stage"
    if st.exists():
        shutil.rmtree(st)
    core = st / "Toby"
    shutil.copytree(OUT / "dist" / "toby", core)
    if (OUT / "dist_cuda" / "toby").exists():   # Toby-full: the CUDA build of the engine (same files otherwise)
        shutil.copytree(OUT / "dist_cuda" / "toby", st / "Toby_cuda")
    shutil.copytree(PUB / "nibbler_scripts", core / "nibbler_scripts")
    shutil.copytree(PUB / "nn", core / "nn")   # weights of the neural move corrector (toby_uci looks for nn/corrector_weights.json next to toby.exe)
    for f in ("README.md", "LICENSE", "NOTICE.md"):
        shutil.copy(PUB / f, core / f)
    shutil.copytree(PUB / "docs", core / "docs")
    # Microsoft Visual C++ 2015-2022 runtime (x64), app-local: on a fresh Windows, torch fails with "[WinError 126] ... c10.dll" without it
    # (Siby's test on Windows Server 2022, 2026-10-31). The redistributable's licence allows copying these DLLs next to the application.
    vc_dst = core / "_internal" / "torch" / "lib"
    vc_dst = vc_dst if vc_dst.exists() else core
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        src = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / dll
        if not src.exists():
            raise SystemExit(f"missing Visual C++ runtime DLL (install the VC++ 2015-2022 x64 redistributable): {src}")
        if not (vc_dst / dll).exists():
            shutil.copy(src, vc_dst / dll)
    if (st / "Toby_cuda").exists():   # the CUDA engine gets the same non-frozen files and runtime DLLs as the CPU one
        cc = st / "Toby_cuda"
        for d in ("nibbler_scripts", "nn", "docs"):
            shutil.copytree(core / d, cc / d)
        for f in ("README.md", "LICENSE", "NOTICE.md"):
            shutil.copy(core / f, cc / f)
        vc2 = cc / "_internal" / "torch" / "lib"
        vc2 = vc2 if vc2.exists() else cc
        for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
            if not (vc2 / dll).exists():
                shutil.copy(vc_dst / dll, vc2 / dll)
    for f in ((MAIA_FT, STOCKFISH) if SHIP_MAIA else (STOCKFISH,)):   # Stockfish: Toby 2's values (without it, network B alone)
        if not f.exists() or f.stat().st_size < 1_000_000:
            raise SystemExit(f"missing or not fetched (Git LFS?): {f}")
    if SHIP_MAIA:
        (st / "maia").mkdir()
        shutil.copy(MAIA_WEIGHTS, st / "maia" / "maia3-79m.pt")
        shutil.copy(MAIA_FT, st / "maia" / "maia3-79m-ft-classical.pt")
    (st / "stockfish").mkdir()
    shutil.copy(STOCKFISH, st / "stockfish" / "stockfish.exe")
    (st / "lc0").mkdir()
    if "lc0.exe" not in LC0_FILES or not NET.exists():   # never ship a Toby-full without Leela (a failed download once produced one silently)
        raise SystemExit(f"lc0.exe or the BT3 network missing in {LC0_DIR}: Toby-full needs both")
    for f in LC0_FILES:
        shutil.copy(LC0_DIR / f, st / "lc0" / f)
    if LC0_GPU_BACKEND:
        (st / "lc0" / "toby_backend.txt").write_text(LC0_GPU_BACKEND + "\n", encoding="utf-8")
    if LC0_CPU_DIR is not None:   # Toby-light: Leela on the CPU (Fast = hybrid path without an NVIDIA card)
        if not LC0_CPU_BACKEND:
            raise SystemExit("TOBY_LC0_CPU_BACKEND must name the backend that passed the policy gate")
        (st / "lc0cpu" / "lc0").mkdir(parents=True)
        for f in sorted(LC0_CPU_DIR.iterdir()):
            if f.is_file() and not f.name.endswith((".pb.gz", ".zip")):
                shutil.copy(f, st / "lc0cpu" / "lc0" / f.name)
        (st / "lc0cpu" / "lc0" / "toby_backend.txt").write_text(LC0_CPU_BACKEND + "\n", encoding="utf-8")
    (st / "nets").mkdir()
    shutil.copy(NET, st / "nets" / NET.name)
    if not TOBY2_WEIGHTS.exists() or TOBY2_WEIGHTS.stat().st_size < 1_000_000:   # never a Toby 2 package without its network
        raise SystemExit(f"Toby 2 weights missing: {TOBY2_WEIGHTS}")
    (st / "toby2" / "nets").mkdir(parents=True)
    shutil.copy(TOBY2_WEIGHTS, st / "toby2" / "nets" / "toby2_b.pt")
    (st / "toby2acc" / "nets").mkdir(parents=True)
    for f, dst in (("toby2_standard.json", "toby2"), ("toby2_level.pt", "toby2"), ("toby2_level_cal.json", "toby2"),   # level head: light and full
                   ("toby2_accurate.json", "toby2acc"), ("toby2_lhead.pt", "toby2acc")):
        src = TOBY2_WEIGHTS.parent / f
        if not src.exists():
            raise SystemExit(f"Toby 2 file missing: {src}")
        shutil.copy(src, st / dst / "nets" / f)
    full = st / "test" / "Toby"          # the complete package as a user would unzip it (used by package_test.py)
    shutil.copytree(st / "Toby_cuda" if (st / "Toby_cuda").exists() else core, full)
    for comp in (("maia",) if SHIP_MAIA else ()) + ("stockfish", "lc0", "nets"):
        shutil.copytree(st / comp, full / comp)
    for d in ("toby2", "toby2acc"):
        for f in (st / d / "nets").iterdir():
            shutil.copy(f, full / "nets" / f.name)
    print("staged:", st)


def toby_version():
    """VERSION of the engine being packaged (public_release/toby_uci.py), written to Toby/VERSION and to the manifest."""
    import re
    return re.search(r'^VERSION = "([^"]+)"', (PUB / "toby_uci.py").read_text(encoding="utf-8"), re.M).group(1)


def zip_all():
    st = OUT / "stage"
    z = OUT / "zips"
    z.mkdir(exist_ok=True)
    # Toby-full: everything (Precision = Accurate uses Leela in Classical with an NVIDIA GPU); Toby-light: no Leela (any PC; Accurate = Fast);
    # Toby-core: the engine alone (bring your own components)
    light_leela = ["lc0cpu/lc0", "nets"] if (st / "lc0cpu" / "lc0").exists() else []   # Toby-light with Leela on the CPU when staged
    variants = {"Toby-core": [], "Toby-light": ["maia", "stockfish", "toby2/nets"], "Toby-full": ["maia", "stockfish", "lc0", "nets", "toby2/nets", "toby2acc/nets"]}
    if not SHIP_MAIA:   # 2.2.1: no Maia, and no Toby-core (it has no Toby 2 network: without Toby 1.0 it could not answer)
        variants = {"Toby-light": ["stockfish", "toby2/nets"], "Toby-full": ["stockfish", "lc0", "nets", "toby2/nets", "toby2acc/nets"]}
    man = {}
    for name, extra in variants.items():
        path = z / f"{name}-{os.environ.get('TOBY_RELEASE_DATE') or __import__('time').strftime('%Y%m%d')}.zip"   # dated: no stale CDN copy (lesson I9)
        path.unlink(missing_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            eng = st / "Toby_cuda" if (name == "Toby-full" and (st / "Toby_cuda").exists()) else st / "Toby"   # Toby-full: Maia on the GPU
            for src in [eng] + [st / e for e in extra]:
                for f in src.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(eng) if src == eng else f.relative_to(src.parent if src.parent != st else st)
                        zf.write(f, str(Path("Toby") / rel))
            zf.writestr("Toby/VERSION", toby_version() + "\n")   # Siby reads it (Toby updates from Siby)
        print(name, f"{path.stat().st_size / 1e6:.0f} MB", flush=True)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        man[name.replace("Toby-", "")] = {"fichier": path.name, "sha256": h.hexdigest(), "taille": path.stat().st_size}
    (z / "manifest_windows.json").write_text(json.dumps({"version": toby_version(), "windows": man}, indent=1))   # merged by make_toby_manifest.py


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2] if len(sys.argv) > 2 else "cpu")
    else:
        {"assemble": assemble, "zip": zip_all}[sys.argv[1]]()
