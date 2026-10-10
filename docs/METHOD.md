# Toby: method page


---

## 1. What "test games" means

* **Test games** are games that were never used to choose the rule, the constants, or any other setting of Toby.
* Before opening a set of test games, the protocol (what is computed, which comparison, which success criteria) was written down, committed to the repository, and tagged. Only then were the games analysed, once.
* Every result is computed game by game: confidence intervals (95%) come from a bootstrap that resamples whole games, not positions.
* Failures are published, with their causes (section 5).

| Test | Tag (protocol / results) | Games | What it tested |
|---|---|---|---|
| Final test | `protocole-final-v1*` / `resultats-final-v1` | 1,500 classical (TWIC); about 430 rapid and 407 blitz Lichess games, plus 300 TWIC games for each of rapid and blitz | Moves and evaluation, three time controls |
| OTB / online validation test | `protocole-lieu-v1` / `resultats-lieu-v1` | 300 TWIC games and 435 / 405 Lichess games (rapid / blitz) | Evaluation only, after separating OTB and online constants |

The games of the final test were used afterwards to fit the OTB / online constants; they are no longer a test set. The validation test used other games.

## 2. The move rule

For every legal move of the position:

1. **Maia-3** (79 M parameters) gives its probability `p` for a player of the given rating. The rating given to Maia is raised in classical games (strength + 700, capped at 2900), and set to the Lichess-equivalent rating (capped at 2800) in rapid and blitz.
2. **Leela** (BT3 network, 1024 nodes) gives the value of each move. The loss of a move is its expected-score difference with Leela's best (most-visited) move, in points. Moves that received fewer than 8 visits get a short extra search (64 nodes). A move that still has no value keeps exactly its Maia probability.
3. Toby computes `score(move) = ln p - lambda x loss/100 + gamma x [best move]`, then normalises with a softmax.
4. For strong players, a "ramp" raises the weight of Leela's best move when Maia gives it very little: below `p1 = 0.08 x (E - 2200) / 500` (capped at 20%) its prior is replaced by `0.8 p1 + 0.2 p`.

| Mode | Rating given to Maia | Severity lambda | Bonus gamma |
|---|---|---|---|
| Classical | E + 700, capped at 2900 | max(0; 3.44 + 5.25 x (E - 2000)/1000) | 0.43 |
| Rapid | Lichess-rapid equivalent of E, capped at 2800 | max(0; (E - 2100) / 72.5) | 0.43 x min(1; lambda) |
| Blitz | Lichess-blitz equivalent of E, capped at 2800 | max(0; (E - 2200) / 98) | 0.26 x min(1; lambda) |

`E` is the strength as a FIDE-equivalent rating (Lichess and Chess.com ratings are converted with the tables of [ChessGoals](https://chessgoals.com/rating-comparison/), a public page that compiles Chess.com and Lichess data; margin of +/- 70 to 115 points). The loss is measured in expected score, never in raw centipawns. Below its threshold each mode plays exactly like Maia.

## 3. The evaluation (win / draw / loss)

* For each move, Toby estimates the win, draw, and loss of **a player of that level** against a player of that level, not of a perfect engine.
* It combines Leela's expected score and Maia's value head (read at a rating of at most 2700) in a Davidson-type three-outcome model: the difference between the two sides sets the win / loss balance and a draw term depends on the strength of the players, on how unbalanced the position is, and on Maia's draw estimate. Equal inputs give 50%.
* Constants are fitted per time control. For rapid and blitz there are separate OTB and online constants (OTB when the rating source is FIDE, online for Lichess and Chess.com; classical is always OTB). When the opponent's rating is given, a second set of constants uses it.
* A move that gives mate is 100% win; a mate forced further away is not (a player can miss it).
* Fit: five-fold cross-validation by game on open data, constants then frozen.

## 4. Data

* **Games:** TWIC (OTB games: classical, rapid, blitz) and Lichess (monthly dumps, non-bot games, CC0). Test games are absent from the tuning data.
* **Positions:** 10 random positions per game, plies 20-99 for the final test. Positions where Leela returned nothing (forced moves) were left out (83 / 32 / 30 in classical / rapid / blitz).
* **Tuning:** open data, about 15,000 classical, 7,100 rapid and 9,500 blitz positions at the first fit of the evaluation, plus the earlier benchmark games.
* **Hardware:** test runs used a rented GPU (RTX 4090 or equivalent) for Leela and an ordinary PC for Maia. In the engine itself, a position takes about 2 to 3 seconds on a laptop with a small GPU; Maia's prediction appears immediately.

## 5. Results

### 5.1 Moves (final test)

| | Top-1: move actually played = most likely move |
|---|---|
| Classical, 1,500 games, 14,917 positions | Maia at real rating 52.4 [51.5; 53.2]; Leela 53.9; Maia with the boosted rating 55.3; **Toby 58.7 [57.9; 59.6]** |
| Rapid, 14,311 positions (E >= 2100) | Maia (boosted) 58.0; **Toby 60.4**; Leela 52.2 |
| Blitz, 12,926 positions (E >= 2200) | Maia (boosted) 60.4; **Toby 61.1**; Leela 49.4 |

* Toby's correction alone (against Maia given the same boosted rating): classical **+3.45 points [+2.92; +3.99]**, rapid **+2.40 [+1.87; +2.96]**, blitz **+0.74 [+0.35; +1.14]**.
* All eight pre-registered classical criteria passed.
* In rapid the pre-registered comparison (Lichess games only) passed: +1.76 [+0.94; +2.60].
* **Failure, blitz:** on the pre-registered Lichess games the gain was +0.05 [-0.64; +0.67] and the criterion "non-inferior to Maia within 0.5 point" failed by 0.14 point. The pooled blitz figure above (all fresh games, OTB and online) is an after-the-fact analysis, not part of the protocol.
* At 2600 and above in classical games, Leela alone does as well as Toby (61.1% against 61.3%); Toby's edge there is over Maia (54.7%).

### 5.2 Move quality

Classes by expected-score loss measured by Leela: good <= 5 points, inaccuracy 5-10, mistake 10-20, blunder > 20. Random samples of the fresh games (500 classical, 300 rapid, 300 blitz games), current engine.

| Blunders (share of moves) | Real players | Maia (boosted rating) | Toby |
|---|---|---|---|
| Classical | 3.5% | 8.9% | 3.5% |
| Rapid | 3.6% | 8.6% | 4.1% |
| Blitz | 5.2% | 8.6% | 5.8% |

| AESL (points per move; measured with Leela's values, so partly circular for Toby) | Real players | Maia | Toby |
|---|---|---|---|
| Classical | 3.04 | 5.84 | 3.11 |
| Rapid | 3.30 | 5.78 | 3.46 |
| Blitz | 3.65 | 5.31 | 3.90 |

Blunders account for 75 / 80 / 86% of the AESL gap between Maia and real players (classical / rapid / blitz).

### 5.3 Evaluation, final test

| Classical, three-outcome log-loss (lower is better) | |
|---|---|
| **Toby** (rating only) | **0.7884** |
| **Toby with the opponent's rating** | **0.7157** |
| Leela alone (raw WDL) | 0.8647 |
| Maia alone (value head) | 1.1210 |
| Constant frequencies | 1.0986 |

* Calibration (mean gap between predicted and actual expected score, in points): Toby 7.4, Leela 27.1, Maia 48.6.
* **Failure, rapid and blitz (Lichess games):** criteria E1, E3, and E5 failed. Rapid log-loss 0.8055 for Toby against 0.8074 for Maia alone; calibration 13.8 points.
* **Cause:** the constants had been fitted on a mix of OTB (TWIC) and online (Lichess) games, but the same strength draws far more often OTB: in balanced positions 38-50% against 17-24% online in rapid, and 24% against 15% in blitz. The model predicted about 33% in rapid, right for neither.

### 5.4 Evaluation, OTB / online validation test (after the fix)

19 of 24 pre-registered criteria passed.

| Log-loss | New constants | Previous constants | Maia alone |
|---|---|---|---|
| Rapid, OTB (300 games) | **0.7832** | 0.7913 | 0.9400 |
| Rapid, online (435 games) | **0.8388** | 0.8693 | 0.8506 |
| Blitz, OTB (300 games) | **0.8141** | 0.8295 | 0.8428 |
| Blitz, online (405 games) | **0.8177** | 0.8227 | 0.8361 |

* **Criteria that failed:** rapid OTB, gain over the previous constants not significant (-0.0080 [-0.0199; +0.0039]); rapid online, not demonstrably better than Maia alone (-0.0118 [-0.0282; +0.0051]); giving the opponent's rating gains little online; blitz online calibration 8.3 points against a bar of 8.
* OTB, the opponent's rating helps a lot: rapid 0.7832 -> 0.7005, blitz 0.8141 -> 0.7535, classical 0.7884 -> 0.7157.

## 6. What was tried and not kept

All on tuning data unless stated; none of these changed the rule.

| Idea | Result |
|---|---|
| Reuse Leela's search tree from one position to the next (faster analysis) | Same top move in 97% of positions, no loss in predicting the real move (log-likelihood -0.0076 [-0.019; +0.003]), but end-to-end time unchanged (a second Leela process slows the first by ~25%). Not adopted. |
| Oracle's mate-in-1 factor, or a flat penalty for moves allowing mate | No gain. Toby already gives such moves 3.3% of the probability, like real players (3.3%). |
| Fading the correction in the opening | Helps classical only after move 30, hurts rapid and blitz. Not adopted. |
| A "puzzle gap" term for positions right after a blunder | No gain on ordinary games. Dropped. |
| Sharing the best-move ramp between moves with almost the same value (weight exp(-loss/tau), tau = 1 to 12 points) | Worse: classical -0.0046 [-0.0071; -0.0023] at tau = 1, worse still at larger tau; rapid -0.0040; blitz neutral. |
| Sharing only the constant bonus gamma between near-equivalent moves | Classical +0.0022 [+0.0006; +0.0039] at tau = 1; nothing in rapid and blitz. Too small and not confirmed on fresh games: not adopted. |
| Sharing both bonuses only between moves tied for Leela's value (<= 1 point) AND Maia's probability (within a factor 1.25-2) | Worse in classical (-0.0011 to -0.0043) and rapid (-0.0007 to -0.0052); neutral in blitz. |
| A Leela search of 4096 nodes instead of 1024 (everything else unchanged; protocol written before the results) | **Moves, 2,937 tuning positions (paired):** log-likelihood +0.004 [-0.004; +0.013], top-1 -0.34 point [-0.87; +0.14]; the pre-registered non-inferiority margin on top-1 (-0.5 point) was not met, so 1024 was kept. The best move changes in 12% of the positions (3% between two 1024-node searches), in both directions. **Final-test positions (paired, same positions at both budgets):** classical (6,000) top-1 57.85% at 1024 against 57.72% at 4096 (-0.13 [-0.53; +0.25]), evaluation log-loss 0.7743 against 0.7757; rapid (3,000) 59.67% against 59.43% (-0.23 [-0.72; +0.26]), 0.8394 against 0.8401; blitz (3,000) 60.90% against 60.90%, 0.8186 against 0.8184. Calibration unchanged. Only Leela's raw win / draw / loss improves (log-loss 0.849 -> 0.844 classical, 1.095 -> 1.071 rapid, 1.263 -> 1.218 blitz): the model's constants were fitted with 1024-node values. No measurable gain on moves or on evaluation: 1024 is kept. |

**What the last tests show:** when two moves have the same value for Leela and the same probability for Maia, players still choose Leela's most-visited move more often than the other: 56% in classical, 65% in rapid, 56% in blitz (clean tied pairs where one of the two was played, n = 559 / 244 / 421; Maia alone would say 50.5%). The move that Leela's search favours carries information that neither the value nor Maia's probability contain, so the bonus is kept on that single move.

**Run-to-run stability.** Because Leela's search is not deterministic, which of two near-tied moves is the most visited can change from one run to the next. We searched 470 classical positions (mover rating 2200+) containing such a near-tie twice, each time from a fresh tree at 1024 nodes. The most-visited move was the same in 82% of the positions (the highest-Q move: 67%). Toby's two distributions differ by 0.028 on average (total variation), by more than 0.2 in 8 positions (1.7%), never by more than 0.4; the probability of the move actually played moves by 1.7 points on average. A 4096-node search did not predict the played move better than 1024 nodes (51% against 48% on 65 positions, within noise), so Toby does not search longer on near-ties. The point of the previous paragraph holds on these new searches too: the move played is the most-visited one more often than the other tied moves (43-46% against 25% if all tied moves were equivalent; 54-59% in the clean pairs, n about 60).

## 7. Limits

* The gains are measured **against Maia**; Toby is not claimed to be the best possible predictor.
* Blitz: +0.7 point on average over the fresh games, not demonstrated on the pre-registered Lichess test.
* Online rapid and blitz evaluation: Maia alone is nearly as well calibrated as Toby.
* Puzzle-like positions (one move much better than the second best, typically right after an opponent's blunder): the evaluation gives the player to move 6 to 8 points less than he really scores (selection effect, no correction).
* Classical online (Lichess classical) is calibrated with the OTB constants; Chess.com has no test games.
* Rating conversion: the source tables have no data below about FIDE 1660. Below that, Toby takes one FIDE point to be one platform point (only used in rapid and blitz with a FIDE rating); nothing was tested there, and in these modes below 2100 / 2200 Toby plays like Maia.
* Opening: no test before move 10; in classical games Toby is slightly below Maia before move 20 (-0.021 and -0.016 in log-likelihood at plies 0-9 and 10-19; not significant).
* 10 random positions per game, not whole games.
* Toby predicts an "average" player of a rating, not a specific person.
* Not fully deterministic, because Leela's search is not: two analyses of the same position differ by 3 points on average in Toby's probabilities (more than 10 points in about 10% of positions), and the win / draw / loss of a move changes by a point or two.
* Near-ties: when two moves are equivalent for Leela, which one receives the bonus can change from one run to the next (it is the more visited one). See section 6.

## 8. Checking

The protocols (written and tagged before the test games were opened), the full result reports including the failed criteria, the scripts and the list of test games are kept in the development repository.
It is not public: it is opened to anyone who wants to check the figures, on request: contact@yoshachess.com.

