# Toby - The Oracle By Yosha: license and third-party components

Toby is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**: see the `LICENSE` file.
Source code: https://github.com/Yosha87/Toby

Toby uses or ships the following components, which have their own licenses:

| Component | License | Source | In which packages, and use |
|---|---|---|---|
| Stockfish 19 (unmodified official build) | GPL v3 | https://stockfishchess.org; source of this version: https://github.com/official-stockfish/Stockfish/tree/sf_19 | all packages (`stockfish/`): value of each move (separate program) |
| Toby 2 network B (`nets/toby2_b.pt`), made for Toby: **modified** Leela Chess Zero network | GPL-3.0-or-later (derivative of the BT4 network below) | fine-tuned from BT4-1024x15x32h-swa-6147500-policytune-332 on over-the-board games, with new rating / time control / clock / venue inputs, a per-move value path reading Stockfish 19 values, and Leela's own policy and value heads; architecture and input encoding are Leela Chess Zero's (lc0 `src/neural/onnx/converter.cc`, `src/neural/encoder.cc`, GPL v3), rebuilt in PyTorch (`toby2net/leela_torch.py`) | all packages: distribution of human moves and the position's win / draw / loss |
| BT4-1024x15x32h-swa-6147500-policytune-332 network (Leela Chess Zero team) | GPL-3.0-or-later (as checked by Yosha, 2026-11-16; the network file's own licence field is empty, as read by `leela_torch.pb_info`) | https://storage.lczero.org/files/networks-contrib/ | the starting point of network B (not shipped as such) |
| Toby 2's level head (`nets/toby2_level.pt`) and, in the full packages, its Leela head (`nets/toby2_lhead.pt`), made for Toby | AGPL v3 (Toby's own files) | trained by Yosha Iglesias for Toby | the game-review level estimate; Accurate's Leela correction |
| Leela Chess Zero, `lc0`, full packages only. Toby-full (Windows): the unmodified official `v0.32.1` `windows-gpu-nvidia-cuda12` release, as published (including `lc0-training-client.exe`, not used by Toby, and the `mimalloc` DLLs). Toby-linux-full: compiled for Toby from the official source (branch `release/0.32`, unmodified, CUDA 12.4) | GPL v3 with an additional permission for the NVIDIA libraries (see `lc0/COPYING` and `lc0/README.txt` in Toby-full) | https://github.com/LeelaChessZero/lc0 (Windows: tag `v0.32.1`; Linux: branch `release/0.32`) | `lc0/`: the Leela search of Accurate (separate program) |
| BT3-768x15x24h-swa-2790000 network (Leela Chess Zero team), full packages only | No explicit license found; distributed for free download on lczero.org, shipped unmodified with its source named | https://lczero.org/play/networks/bestnets/ | `nets/`: the network `lc0` evaluates positions with |
| mimalloc (Microsoft), Toby-full (Windows) only, part of the official `lc0` release | MIT | https://github.com/microsoft/mimalloc | memory allocator of `lc0` |
| NVIDIA CUDA libraries, full packages only: with `lc0` in Toby-full (Windows: cudart, cuBLAS, cuBLASLt 12.9, see `lc0/CUDA.txt`); inside the frozen application with PyTorch's CUDA build (CUDA runtime, cuBLAS, cuBLASLt, cuDNN 9, cuSPARSE, NVRTC, nvJitLink and, depending on the package, cuFFT, cuSOLVER, cuRAND, CUPTI); in Toby-linux-full, `lc0` uses the cudart / cuBLAS / cuBLASLt copies at the top of `_internal/` | NVIDIA CUDA Toolkit and cuDNN licence terms: redistributable as part of an application, not individually | https://developer.nvidia.com/cuda-toolkit, https://developer.nvidia.com/cudnn | GPU computation for `lc0` and for Toby 2's network |
| PyTorch, NumPy (and SciPy in the Windows packages) | BSD-3-Clause | https://pytorch.org, https://numpy.org, https://scipy.org | inside the frozen application (`toby` / `toby.exe` and `_internal/`): Toby 2's network |
| python-chess | GPL v3 or later | https://github.com/niklasf/python-chess | inside the frozen application: chess rules and move generation |
| Python and the other Python packages bundled by PyInstaller (their licence files are in `_internal/*.dist-info`) | PSF License and their own licences | https://www.python.org, https://pyinstaller.org | inside the frozen application |
| Maia-3 code (University of Toronto CSSLab) and its dependencies huggingface_hub and hf_xet: frozen into the application by the build, **not used by Toby 2** (no Maia weights are shipped, Toby 1.0 is not part of these packages) | Maia-3: AGPL v3; huggingface_hub, hf_xet: Apache-2.0 | https://github.com/CSSLab/maia3, https://github.com/huggingface/huggingface_hub, https://github.com/huggingface/xet-core | inside the frozen application only |

Rating conversion tables come from ChessGoals (https://chessgoals.com/rating-comparison/), a public page that itself compiles public Chess.com and Lichess data. The values were copied from that page and the source is credited.


## Additional terms (section 7 of the GNU AGPL v3)

These terms apply to Toby's own code and files (copyright © 2026 Yosha Iglesias), not to the third-party components listed above.

1. **Attribution (section 7(b)).** Every copy and every modified or renamed version of Toby, and every work that contains it, must preserve the
   author attribution "Toby © Yosha Iglesias, toby.tribychess.org" in its Appropriate Legal Notices: in this NOTICE file or an
   equivalent notice shipped with it, and wherever the work shows its legal notices or credits to the user (for example an "About" box, a
   start-up banner, or the engine's UCI `id author` line).
2. **Modified versions (section 7(c)).** A modified version must be marked in reasonable ways as different from the original version, and
   must not be presented as Toby itself or as made or endorsed by Yosha Iglesias.
3. **Names (section 7(e)).** The AGPL does not grant any right to use the names "Toby", "The Oracle By Yosha" or "Yosha" as trade names,
   trademarks or service marks, except as required to give the attribution of point 1.

This note is not legal advice.
