<p align="center"><img src="docs/icon.png" width="160" alt="Toby"></p>

# Toby - The Oracle By Yosha

**Toby 2. Better. Faster. Lighter.** The best human-like chess engine, free and open source.

Toby predicts the moves humans play, from beginner to 2900, in classical, rapid and blitz, and what each move is worth for them: win,
draw or loss for players of that level, against that opponent. A UCI engine for ChessBase, Nibbler or any interface that shows several
lines, and best used with Siby.

Toby 1.0 was already the state of the art in human-like chess. Toby 2 is significantly better at every level, from beginners to 2700+.
Leela, fine-tuned on real human games, gives the intuition; Stockfish does the calculation. System 1 and System 2, like a human.

## Centipawns suck: Toby gives useful evaluations

A position is only ever won, drawn or lost. Every other evaluation is practical: true for some players, in some conditions. Stockfish's
evaluations are true for Stockfish against itself: it ignores who is playing, the clock and the risk of a blunder. Toby gives the chances
of winning, drawing and losing for the two players of the game, from their ratings, the time control, over the board or online, and the
moves they are likely to play, blunders included.

## How it works

> **Human thinking = intuition + analysis = System 1 + System 2**

* **Intuition:** a neural network fine-tuned from Leela Chess Zero on real over-the-board games predicts the moves humans play and the
  chances of winning, drawing and losing, for the players' ratings and the time control.
* **Calculation:** Stockfish 19 computes the value of each candidate move; Toby uses it to correct the network, for example to keep
  blunders as frequent as in real games.
* **Accurate** (with an NVIDIA graphics card, classical games): a Leela search adds a second look at each move.

Toby is the full version of [Oracle](https://github.com/Yosha87/Oracle) (2024), the proof of concept of this approach. *Yosha Iglesias*

## Results

On 2,201 over-the-board classical games kept out of training, Toby 2 finds the move actually played 65.8 % of the time, Toby 1.0
64.7 %, Maia-3 58.7 %, and Toby 2 is ahead at every level, from beginners to 2700+. Details: https://toby.tribychess.org

## Download

**Toby 2, version 2.0** (November 2026): https://toby.tribychess.org

| Package | For |
|---|---|
| Toby-light (0.63 GB) | Windows 10 / 11, 64-bit, any computer |
| Toby-full (3.9 GB) | Windows with an NVIDIA graphics card (Accurate) |
| Toby-linux-light (0.69 GB) | Linux 64-bit (Ubuntu 22.04 or newer, glibc 2.35+), any computer |
| Toby-linux-full (3.7 GB) | Linux with an NVIDIA graphics card (Accurate on RTX 30xx / 40xx, A40, A5000, L4) |

Unzip, then add `Toby/toby.exe` (Linux: `Toby/toby`) as a UCI engine. With Siby: Tools > Add an engine > Toby. Checksums:
[toby_manifest.json](https://toby.tribychess.org/toby_manifest.json).

## Speed

* With an NVIDIA graphics card, in Accurate: final analysis of a position in about 0.7 s (NVIDIA T4), up to 3.7 times faster than Toby 1.0.
* On the processor alone: about 1 s per position; a first, provisional answer from the network in about 0.2 s.
* The first answer after starting takes a few seconds more.

## Options

| Option | Meaning |
|---|---|
| `White Rating`, `Black Rating` | The ratings of the two players. |
| `Rating Source` | FIDE (over the board), Lichess or Chess.com (online): Classical, Rapid or Blitz lists. |
| `Time Control` | Classical, Rapid or Blitz. |
| `Base Time`, `Increment` | The game's time control in seconds (base time, increment); 0 / -1 (default): unknown, or the increment taken from the `go` command. |
| `Toby 2 Product` | Standard (any computer) or Accurate (NVIDIA graphics card, classical; listed only where it can run). |
| `Network Device` | Where the network runs: Auto (default: the product's choice), CPU or GPU. |
| `Network Threads`, `Stockfish Threads` | Processor threads of the network and of Stockfish; 0 (default): automatic. |
| `Move Evaluations` | Toby 2 (default): the win / draw / loss of each move, for these players; Off. |
| `Position Evaluation` | Toby 2 (default): the position's win / draw / loss from the likely moves; Network: the network's own estimate. |
| `Deepening` | On (default): in infinite analysis, refines the win / draw / loss of the most likely moves. |
| `Provisional Lines` | On (default): a first answer from the network, then the full answer (not under a game clock). |
| `Leela Head` | Accurate only: Product (default: on in classical), On or Off. |
| `Display` | ChessBase, Nibbler, Standard or Full (for interfaces built for Toby, such as Siby); detected automatically. |
| `MultiPV`, `UCI_ShowWDL` | Number of lines shown (default 500: every listed move) and the win / draw / loss in each line. |

## Limits

* Win / draw / loss: in classical, on average about 2 points too optimistic for the side to move from 2250 up; online rapid and blitz
  predict too many draws. Toby's Stockfish budget can miss a very deep win or a fortress.
* Blunders: in classical, Toby 2 blunders a little less than real players below about 2100, and the correction is slightly too strong
  between 2100 and 2600.
* Accurate needs an NVIDIA graphics card. On Linux it runs on RTX 30xx / 40xx, A40, A5000 and L4 cards; on other cards Toby says so and uses
  Standard.
* Lichess and Chess.com ratings are converted to FIDE with public tables.
* The level estimate of a game review is a fun tool: with the players' ratings it mostly reflects them; without, one game cannot tell
  levels above about 2500 apart.

## Triby

**Siby**, my free chess GUI built around Toby, and **Gaby**, a database of more than 13 million games, are available at
[tribychess.org](https://tribychess.org). Toby, Siby and Gaby: **Triby**, free, open source and without ads.

## Quick start

Unzip a package, then add `toby.exe` (in the `Toby` folder; Linux: `toby`) as a UCI engine in your interface.

**ChessBase:** Engines > Create UCI engine > choose Toby; open its options and set White Rating, Black Rating, Rating Source, Time Control and Display = ChessBase. Start an analysis and wait a few seconds.
In the lines, the score column is the percentage of players of that rating who play the move, and the bar / numbers are its win, draw and loss. When it is Black's turn ChessBase shows the numbers with a minus sign: read the absolute value.

**Nibbler:** Engine > Choose engine > Toby (Toby detects Nibbler by itself). Each line reads `52.9%  d4  (N: 45.67%, P: 34.43%, WDL: 436 235 329)`: expected result of the move, `N` = Toby's probability, `P` = the network's probability before Toby's correction by the move values, `WDL` = win / draw / loss in per mille.
Copy the files of `nibbler_scripts` (Windows packages) into Nibbler's scripts folder (menu Engine > Custom scripts) to set the time control and the ratings in three clicks.

While it thinks: with Provisional Lines on, the network's first answer appears at once, then Toby's final answer.

## Licence and credits

* Toby: **AGPL v3** (`LICENSE`, `NOTICE.md`).
* Shipped in the packages, with their own licences (details in `NOTICE.md`): Stockfish 19 (GPL v3); Toby 2's network, a modified Leela
  Chess Zero network (GPL v3 or later); in the full packages, Leela Chess Zero `lc0` v0.32.1 with the BT3 network and the NVIDIA CUDA
  libraries; PyTorch and NumPy (BSD-3-Clause), python-chess (GPL v3).
* Rating tables: [ChessGoals](https://chessgoals.com/rating-comparison/). Games: The Week in Chess and Lichess (CC0).

## Support

Contributions are welcome: code, games, tests in other interfaces. If you value this work, please consider
[a donation](https://tribychess.org/support).
