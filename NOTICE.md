# Toby - The Oracle By Yosha: license and third-party components

Toby is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**: see the `LICENSE` file.
Source code: https://github.com/Yosha87/Toby

Toby uses or ships the following components, which have their own licenses:

| Component | License | Source | Use |
|---|---|---|---|
| Maia-3 (code and the 79M weights, University of Toronto CSSLab) | AGPL v3 (repository `LICENSE` and the model card) | code: https://github.com/CSSLab/maia3 (version shipped: commit 1e13597c42d4858b7cfd7cfdae01e297263364b2); weights: https://huggingface.co/UofTCSSLab/Maia3-79M (snapshot a107d6ceb7b298cb04ae1da4edffe2939858b894) | distribution of human moves, value head |
| Fine-tuned Maia-3 weights for Classical (`maia/maia3-79m-ft-classical.pt`), made for Toby: **modified** weights | AGPL v3 (derivative of the Maia-3 weights) | fine-tuned from Maia-3 79M on 1.3 million over-the-board classical games (Lichess broadcast database 2020-2026 with the time control measured from the clocks, and The Week in Chess 2014-2026) | distribution of human moves in Classical |
| Leela Chess Zero, `lc0` v0.32.1 (unmodified official CUDA 12 build; Toby-full) | GPL v3 with an additional permission for the NVIDIA libraries | source of this version: https://github.com/LeelaChessZero/lc0/tree/v0.32.1 | value of each move (separate program) |
| Leela Chess Zero, `lc0` v0.32.1 for the processor: Toby-light (Windows) ships the unmodified official `onnx-dml` release, run with the backend `onnx-cpu`; Toby-linux-light and Toby-linux-full ship `lc0` compiled from the official source with a small build patch (`lc0/lc0-v0.32.1-build.patch`, shipped in the package) | GPL v3 with an additional permission for the NVIDIA libraries | source of this version: https://github.com/LeelaChessZero/lc0/tree/v0.32.1 | value of each move without an NVIDIA GPU (separate program) |
| ONNX Runtime (Microsoft), shipped next to the processor `lc0`: 1.22 in Toby-light (Windows, part of the official `lc0` release), 1.20.1 in the Linux packages | MIT | https://github.com/microsoft/onnxruntime | neural network computation for the processor `lc0` |
| NVIDIA CUDA libraries shipped with `lc0` (cuBLAS, cuBLASLt, cudart) | NVIDIA CUDA license, section 2.2 (see `lc0/CUDA.txt` and `lc0/README.txt`): redistributable together with the `lc0` package as a whole, not individually | https://developer.nvidia.com/cuda-toolkit | GPU computation for `lc0` |
| BT3-768x15x24h-swa-2790000 network (Leela Chess Zero team) | No explicit license found; distributed for free download on lczero.org, shipped unmodified with its source named | https://lczero.org/play/networks/bestnets/ | the network `lc0` evaluates positions with |
| PyTorch, NumPy | BSD-3-Clause | https://pytorch.org, https://numpy.org | inside the frozen application (Maia) |
| python-chess | GPL v3 or later | https://github.com/niklasf/python-chess | chess rules and move generation |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub | model loading (Maia) |
| Stockfish 19 (unmodified official build) | GPL v3 | https://stockfishchess.org; source of this version: https://github.com/official-stockfish/Stockfish/tree/sf_19 | value of each move (separate program): the calculator of `Precision = Fast`, and of `Accurate` outside Classical with an NVIDIA GPU; also the values of the moves Leela barely explored |

Rating conversion tables come from ChessGoals (https://chessgoals.com/rating-comparison/), a public page that itself compiles public Chess.com and Lichess data. The values were copied from that page and the source is credited.

Maia-3 is described in: Monroe, Eilender, Chalmers, Tang, Anderson, *Chessformer: A Unified Architecture for Chess Modeling*, ICLR 2026.

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
