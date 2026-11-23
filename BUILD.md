# Building Toby 2.0

How the four Toby 2.0 release packages are built from this source. The packages: Toby-light and Toby-full for Windows, Toby-linux-light and
Toby-linux-full.

## Inputs

**Engine code** (this repository, frozen into `toby` / `toby.exe` by PyInstaller):
- `toby_uci.py` (the UCI engine), `toby2.py`;
- `toby2net/` (`model.py`, `leela_torch.py`, `encode.py`, `leela_feats.py`, `levelgen.py`, `levelhead.py`);
- `corrector.py`, `wdl_endgame_p1.py`, `toby_rating_convert.py`.

**Maia-3's code**, https://github.com/CSSLab/maia3, commit `1e13597c42d4858b7cfd7cfdae01e297263364b2`.
- It is frozen into the application by the build. Toby 2 does not use it: no Maia weights are shipped.
- It comes with its Python dependencies (`huggingface_hub`, `hf_xet`).

**Networks and data**, shipped in `nets/`, checked by sha256 when building. The small JSON files are in `nets/` of this repository; the
`.pt` / `.pb.gz` files are in the release packages.

| File in the package | sha256 |
|---|---|
| `toby2_b.pt` (network B) | `3219cfc6f01bcd218441f67466475f27b05174fd462c40c8a4053d9753a1b281` |
| `toby2_standard.json` | `395670f85615c0f638c9284e1115a8579fac979d5d7b477a6257147a2f17a1a4` |
| `toby2_accurate.json` (full packages) | `b1b896986eb14f3bfcea20e77be7da1de10df2b3897138a3a62768e4424210be` |
| `toby2_lhead.pt` (full packages) | `d0c278bba6b87369f79e1cd8d89ba11227fdfd173734aeba52f001d2e78497da` |
| `toby2_level.pt` | `6e83ce50c7d688847f5ad8be5c855c2638b6b678004b617ec70e8a639bb7f6ce` |
| `toby2_level_cal.json` | `1a104601c5b093ec021d3eb63c057d1206c8f36fc7f1e040765ef56f2efd1e22` |
| `BT3-768x15x24h-swa-2790000.pb.gz` (full packages) | `e3067757d1fc2dfc66947b21d15ace0cedf4c54254fc1de83d77c378a3e8b8e1` |

`toby2_level_cal.json` is made by `build/merge_ea_display.py` from `nets/toby2_level_calibration_base.json` and
`nets/toby2_level_ea_display_fields.json`. The script checks both inputs by sha256, and its output is reproducible.

**Programs shipped next to the engine:**
- **Stockfish 19:** the official `stockfish-linux-x86-64-universal` / Windows release. Checked with `stockfish bench` = 2497913 nodes.
- **`lc0`, full packages only:**
  - **Windows:** the official `lc0-v0.32.1-windows-gpu-nvidia-cuda12` release, unchanged (`lc0.exe` sha256 `e32164ce...`).
  - **Linux:** built from https://github.com/LeelaChessZero/lc0, branch `release/0.32` (the binary reports `Lc0 v0.32.1`, built 2026-11-08), with its own `./build.sh release` and CUDA 12.4, unmodified (sha256 `b2a4bb76...`; `toby2_accurate.json` pins it).

## Linux (`build/build_package_linux.py`)

**Build system:** Ubuntu 22.04 (glibc 2.35), Python 3.11, in a virtual environment.
- light: `torch==2.4.1` from https://download.pytorch.org/whl/cpu;
- full: `torch==2.4.1` from `.../whl/cu124`;
- both: `chess==1.11.2`, `numpy`, `huggingface-hub>=0.23` and Maia-3's `requirements.txt`, then `pyinstaller`. PyInstaller 6 was not pinned: it was the latest on PyPI on 2026-11-22. The frozen binary records no version.
- Build on the oldest glibc you want to support. The frozen libraries need at most `GLIBC_2.35`.

**Steps:**
1. The build scripts read this repository's tree from a folder `public_release/` next to them. For example, from the repository's root:
   `ln -s .. build/public_release` (Windows: a copy).
2. `python build_package_linux.py freeze` (light, CPU PyTorch) and/or `python build_package_linux.py freeze cuda` (full, CUDA PyTorch).
   - `freeze cuda` keeps `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12` at the top of `_internal/`, where `lc0` finds them.
   - Environment:
     - `TOBY_MAIA_REPO` (the Maia-3 clone above);
     - `TOBY_STOCKFISH`;
     - `TOBY2_WEIGHTS` (the folder of the files in the table, given by `toby2_b.pt`);
     - full only: `TOBY_LC0_DIR_LINUX` (the folder with `lc0`) and `TOBY_NET` (the BT3 file).
3. `python build_package_linux.py assemble`, then `zip`. The zips and `manifest_linux.json` are written to `package_build_linux/zips/`.

## Windows (`build/build_toby2_windows.ps1`, which calls `build/build_package.py`)

**Build host:** Windows Server 2022 x64, with Python 3.12.8 (python.org installer) and the Visual C++ 2015-2022 x64 runtime.
- **Toby-light:** `torch==2.14.1+cpu` (index https://download.pytorch.org/whl/cpu).
- **Toby-full:** `torch==2.14.1+cu126` (index `.../whl/cu126`).
- **Both:** PyInstaller 6.22.3 with the contrib hooks 2026.10, numpy 2.5.5, scipy 1.18.1, chess 1.11.2, huggingface_hub 2.4.0. Also frozen as
  dependencies: click 8.5.1, httpx2 / httpcore2 2.16.2, markupsafe 3.0.3, tqdm 4.70.1, hf_xet.
- **Maia-3's code:** the same commit as above.
- **Stockfish 19:** `stockfish-windows-x86-64-universal.exe` from the `sf_19` release zip, sha256
  `45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0`.
- **Toby-full only:** the official `lc0-v0.32.1-windows-gpu-nvidia-cuda12.zip` (`lc0.exe` sha256 `e32164ce...`, cudart 12.9) and BT3 from
  storage.lczero.org.

The script downloads these, checks every sha256, reads the six Toby 2 files from a local folder (`-Nets`), and runs
`build_package.py`. Environment: `TOBY_MAIA_REPO_WIN`, `TOBY_LC0_DIR_WIN`, `TOBY_STOCKFISH_WIN`, `TOBY2_WEIGHTS_WIN`, `TOBY_RELEASE_DATE`
(`TOBY_SHIP_MAIA` unset).

## Checking a build

**What a build should give:**
- `uci` prints `info string toby version 2.0`;
- `isready` answers `readyok`;
- a `go movetime 5000` from any position ends with a legal `bestmove`.

**Full packages with an NVIDIA graphics card:**
- `setoption name Toby 2 Product value Accurate`, then `isready`, starts `lc0`;
- its self-check (three fixed positions) must pass before Accurate answers.
