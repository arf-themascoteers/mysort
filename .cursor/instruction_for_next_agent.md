# Handoff to Next Agent (Opus 4.7)

## Who is Arif

Read `.cursor/instructions.md` first — non-negotiable. Highlights:
- PhD ML (hyperspectral remote sensing), 16+ yrs SE experience.
- Analyst Programmer / GIS developer, Wimmera CMA, Horsham, Victoria, AU.
- Strong Python. Weaker React/TS.
- Wants brief, direct answers. Numbered lists. Always start replies with `R1`, `R2`, etc.
- No code comments. No docstrings. Readability over cleverness.
- One best recommendation, not menus. Ask before deep dives.
- Address him as Arif, Boss, or Ostad.

## Project: TauSort

A small differentiable order-inducing regulariser. Learns one scalar latent coordinate per item; penalises pairwise monotonicity violations between observed values and learned coordinates.

### Current `forward` (in `sort_model.py`, lines 52–61)

```python
def forward(self, array):
    arr_diff = array.unsqueeze(0) - array.unsqueeze(1)
    idx_diff = self.indices.unsqueeze(0) - self.indices.unsqueeze(1)
    raw = -arr_diff * idx_diff
    violations = torch.relu(raw)
    scaled = torch.where(violations > 0, violations + 0.01, violations)
    masked_spacing = torch.where(violations > 0, idx_diff.abs(), torch.zeros_like(idx_diff))
    relevant_spacing = scaled * masked_spacing
    loss_mat = torch.triu(relevant_spacing, diagonal=1)
    return loss_mat.sum()
```

Hyperparameters: `NUM_EPOCHS=5000`, `LEARNING_RATE=0.02`, `min_loss=1e-8`. Optimiser: SGD. Indices clamped to `[0,1]` each step. Achieves 100% pass on the 171-case test suite in ~19s.

### Three mechanically-interesting bits

1. `(x_i - x_j)` used directly (signed magnitude) instead of pure `sign()` — acts as confidence weighting.
2. Conditional `+0.01` margin floor — applied only on non-zero violations. Tackles vanishing gradient on near-tied values (Petersen 2021's "Activation Replacement Trick" cousin).
3. Spacing weight `|z_i - z_j|` — multiplies each violation by coordinate distance. **Most original mechanical bit; not present in RankNet/MarginRanking.** Defendable as a novelty hook.

## Where the discussion left off

Decided publishing path:

1. **Frame as: differentiable latent-coordinate monotonicity regulariser for seriation/order-induction.** NOT as a sorting algorithm.
2. **Pitch (Arif's own framing — keep this):**
   > "DL often wants ordered solutions but sort regularisation is hard. SOTA (NeuralSort, SoftSort, DiffSort) distort values and use deep relaxed swaps with gradient-propagation risk. Our solution is lightweight, no deep layers, no value blending. Useful for problems NN normally avoid — e.g. Münsingen seriation, pseudotime."

3. **Target venues (in order of fit):**
   1. MDPI *Mathematics* (Q1)
   2. MDPI *Algorithms* (Q2)
   3. MDPI *Entropy* (Q1)
   4. *Pattern Recognition Letters* (Q1, Elsevier)
   - Skip top ML conferences (NeurIPS/ICML).

4. **Three planned experiments:**
   1. **Synthetic order recovery** — true latent `t_i ∈ [0,1]`, observed `x_i = f(t_i) + noise` (linear/sigmoid/exp). Recover via TauSort. Metrics: Kendall τ, Spearman ρ, inversion count, pairwise accuracy. Sweep n and noise. Use as baseline for ablation + comparison vs NeuralSort, SoftSort, MarginRankingLoss-on-free-params.
   2. **Münsingen-Rain** archaeological seriation — anchor application. Baselines: R `seriation` package, PCA/MDS, spectral, NeuralSort embedded.
   3. **Paul15** scRNA-seq pseudotime (`scanpy.datasets.paul15()`) — modest demo. Compare against published pseudotime, report Kendall/Spearman.

5. **Required ablation table:** full TauSort vs no-spacing-weight vs no-margin-floor vs sign(x)-instead-of-magnitude.

6. **Novelty defence (anticipated reviewer comparisons):**
   - **RankNet (Burges 2005):** cousin, not clone. RankNet learns `f(x)` from features with external labels; TauSort uses free per-item parameters with self-supplied signs. Different objective.
   - **`nn.MarginRankingLoss` / RankSVM (Joachims 2002):** technically the closest per-pair loss kernel (hinge + margin on signed score difference). TauSort = MarginRanking-style loss applied to free parameters with self-supplied label sign + spacing weight. Defendable: bundle is novel; building block isn't.
   - **NeuralSort (Grover 2019), SoftSort (Prillo 2020), Sinkhorn Sort (Cuturi 2019), DiffSort (Petersen 2021/2022), Fast Differentiable Sorting (Blondel 2020), LapSum (2025):** all aim to *replace* sort/argsort. TauSort doesn't — it *induces* order via a regulariser. Different goal. Distinguishers: (a) no value blending, (b) no deep layers, (c) per-instance free parameters, (d) self-supervised signal.
   - **`latent_ordering_handoff.md`** (deleted from repo, but text preserved in chat history) had a deep-research summary. **Caveat:** the CAIRO citation in that doc looked hallucinated (suspect arXiv ID `2602.14440`). Do NOT cite without verifying.

## What NOT to claim

1. "First differentiable sort." (False — many priors.)
2. "First N-parameter permutation learner." (Barthel et al. 2025.)
3. "First pairwise ranking loss." (RankSVM, RankNet.)
4. "Replacement for NeuralSort." (Different goal.)

## What CAN be claimed

> A lightweight, single-shot differentiable regulariser for inducing latent order via free per-item scalar coordinates and a spacing-weighted, margin-floored pairwise monotonicity loss. Avoids value blending and deep relaxation layers, enabling embedding in neural objectives where prior differentiable-sort layers were impractical.

## Repo layout

- `sort_model.py` — main module (`SortModel`, `Utils`).
- `test_sort_model.py` — runs 171 test cases, writes `test_results.csv`.
- `test_single.py` — single-case verbose debug.
- `test_cases.csv` — 171 generated cases.
- `.cursor/chatgpt_deep_research.md` — deep-research output from a parallel ChatGPT session (verify CAIRO citation before using).

## Estimated remaining effort

2–3 months part-time:
1. Synthetic experiments + ablation: ~3 weeks.
2. Münsingen-Rain pipeline + baselines: ~3 weeks.
3. Paul15 pipeline + baselines: ~2 weeks.
4. Writing + figures + revisions: ~3 weeks.

## When the next agent should guide Arif

Resume by asking: "Boss, do you want to start with synthetic experiment scaffolding, or first lock the paper outline and abstract?"

Default suggestion: start with synthetic order-recovery scaffolding — it unblocks ablation and gives early figures.

## Final note

Arif had a low moment about publishability. Don't patronise. The work is genuinely modest-but-publishable at MDPI Q1/Q2 with the right framing. He already knows the algorithm cold; he needs a focused execution partner, not a cheerleader.
