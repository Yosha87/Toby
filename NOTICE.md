# Toby - The Oracle By Yosha: license and third-party components

Toby is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**: see the `LICENSE` file.
Source code: https://github.com/Yosha87/Toby

Toby uses or ships the following components, which have their own licenses:

| Component | License | Source | Use |
|---|---|---|---|
| Maia-3 (code and the 79M weights, University of Toronto CSSLab) | AGPL v3 (repository `LICENSE` and the model card) | https://github.com/CSSLab/maia3 and https://huggingface.co/UofTCSSLab/Maia3-79M | distribution of human moves, value head |
| Leela Chess Zero, `lc0` v0.32.1 (unmodified official CUDA 12 build) | GPL v3 with an additional permission for the NVIDIA libraries | https://github.com/LeelaChessZero/lc0 | value of each move (separate program) |
| NVIDIA CUDA libraries shipped with `lc0` (cuBLAS, cuBLASLt, cudart) | NVIDIA CUDA license, section 2.2 (see `lc0/CUDA.txt` and `lc0/README.txt`): redistributable together with the `lc0` package as a whole, not individually | https://developer.nvidia.com/cuda-toolkit | GPU computation for `lc0` |
| BT3-768x15x24h-swa-2790000 network (Leela Chess Zero team) | No explicit license found; distributed for free download on lczero.org, shipped unmodified with its source named | https://lczero.org/play/networks/bestnets/ | the network `lc0` evaluates positions with |
| PyTorch, NumPy | BSD-3-Clause | https://pytorch.org, https://numpy.org | inside the frozen application (Maia) |
| python-chess | GPL v3 or later | https://github.com/niklasf/python-chess | chess rules and move generation |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub | model loading (Maia) |
| Stockfish | GPL v3 | https://stockfishchess.org | only used in the benchmarks for comparison (separate program); Toby itself does not need it |

Rating conversion tables come from ChessGoals (https://chessgoals.com/rating-comparison/), a public page that itself compiles public Chess.com and Lichess data. The values were copied from that page and the source is credited.

Maia-3 is described in: Monroe, Eilender, Chalmers, Tang, Anderson, *Chessformer: A Unified Architecture for Chess Modeling*, ICLR 2026.

This note is not legal advice.
