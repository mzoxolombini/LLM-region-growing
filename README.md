# LLM-region-growing

Code, designs, prompt and reply logs, and results for the paper *LLM-Driven Automated Design of Region-Based Hyperspectral Image Classification from a Region-Growing Seed* (M. Mbini and T. Nyathi, University of Pretoria).

## Reproducing the paper

| Step | Command | Output |
|---|---|---|
| Data | `python scripts/download_data.py` (Houston 2013 by hand) | `data/` |
| Search (40 designs) | `python scripts/run_search.py --config config_full.yaml` | `runs/full/designs`, `runs/full/manual` |
| DoE and feature subsets | `python scripts/precompute.py --config config_rev.yaml --doe --fs` | `runs/full/doe`, `runs/full/fs_masks_b` |
| Random and 16x16 tile protocols | `python scripts/final_eval_cached.py --config config_full.yaml` | `runs/full/final/jobs` |
| Buffered, low-label, new competitors | `python scripts/final_eval_v2.py --config config_rev.yaml --group main` (then `--group fs`, `--group rev2`) | `runs/full/final/jobs_v2`, `jobs_v3` |
| Tables and tests | `python scripts/summarise_v2.py` | `paper_tables/` |
| Search figure | `python scripts/make_figures_full.py` | `paper_figures/` |

The reported run used `llm.provider: manual`: each prompt was written to `runs/full/manual/gGG_kKK.prompt.md` and the reply read from `gGG_kKK.reply.md`, so all 40 prompts and replies are in the repository. `runs/main` is the two-generation pilot run described in Section 5.4 of the paper. Per-split results, including test-pixel predictions, are stored as gzipped JSON in `runs/full/final/jobs*/`.

## Background (original framework notes)

Code for the experiment in Chapter 8: an LLM rewrites the MRG-ASVM region-growing
pipeline as Python code inside an elitist (mu + lambda) loop, every design is scored
on five hyperspectral scenes with the leakage-controlled fitness of Eq. (8.1), and
the best MRG-family design (LLM-MRG) and the best overall design (LLM-RB) are then
tested on the held-out 30% split against Baseline MRG, MRG-PSO, MRG-ASVM and a DoE
competitor.

## Where each part of the chapter lives

| Chapter | File |
|---|---|
| 8.3 baseline, seed design D0 | `designs/d0_baseline_mrg.py`, `llm_mrg/blocks.py` (`mrg`, `glcm88`) |
| 8.4.1 design contract, ctx | `llm_mrg/context.py`, `llm_mrg/sandbox.py` (`check_contract`) |
| 8.4.2 building blocks | `llm_mrg/blocks.py` |
| 8.4.3 loop, E1/E2/M1/M2, rank selection | `llm_mrg/evolve.py` |
| 8.4.4 prompts and feedback, full logging | `llm_mrg/prompts.py`, `llm_mrg/llm_client.py` |
| 8.4.5 fitness, OA_val, OA_grp, mF1_val | `llm_mrg/harness.py` (`score_search`) |
| 8.4.6 isolation, time limit, failure = 0 | `llm_mrg/sandbox.py` |
| 8.1 choice of LLM-MRG vs LLM-RB | `llm_mrg/select.py` |
| 8.5 test metrics, multi-split, spatial check | `llm_mrg/harness.py` (`score_test`), `scripts/run_final_eval.py` |
| 8.5 statistical tests | `llm_mrg/stats.py` |
| 6.3.3 MRG-PSO, MRG-ASVM (GOA) | `llm_mrg/feature_selection.py` |
| 5.2.3 DoE competitor | `llm_mrg/doe.py` |
| 8.5.1 harness consistency check | `scripts/check_baseline.py` |

## Running it

```bash
pip install -r requirements.txt

# 0. pipeline check, no data or API key needed (about 2 minutes)
python scripts/smoke_test.py

# 1. data: four scenes download from EHU, Houston goes in data/ by hand
python scripts/download_data.py
#    data/Houston.mat and data/Houston_gt.mat  (IEEE GRSS DFC 2013)

# 2. put the real MRG settings into designs/d0_baseline_mrg.py, then check them
python scripts/check_baseline.py --scene indian_pines     # should print 89.82

# 3. the search
export ANTHROPIC_API_KEY=...
python scripts/run_search.py --config config.yaml

# 4. the test-set evaluation
python scripts/run_final_eval.py --config config.yaml
```

Any config value can be overridden from the command line, for example
`--set search.mu=6 llm.provider=openai llm.base_url=http://localhost:8000/v1 llm.model=qwen2.5-coder-32b`.
A run resumes from its last finished generation.

## Outputs

`runs/<name>/` holds `llm_log.jsonl` (every prompt and reply), `designs/*.py` and
`designs/*.json` (code, fitness, per-scene feedback, building-block trace, lineage),
`population_gen*.json`, `selection.json`, and `final/` with `summary.csv`,
`results.json` (per-class accuracy, purity, region counts) and `tests.json`
(McNemar, Wilcoxon, Friedman).

`summary.csv` has a column the chapter doesn't mention: `OA_no_shared_region_mean`, the
test OA on test pixels whose region holds no training pixel. Under region labelling,
test pixels that share a region with training pixels inherit their label, so this
column shows how much of the test OA comes from that.

## What I had to assume

The chapter points to sections that weren't in the file (5.2.3, 5.3, 5.5.x, 6.3,
Table 5.5, Table 6.2, Eqs. 5.7, 6.5, 6.6). Everything below is my reconstruction and
is marked `ASSUMED` or `SET` in the code.

* **V channel**: PC1 to PC3 min-max scaled, read as RGB, V = max(R, G, B).
* **MRG growth rule**: per 3x3 cell, pixels ordered by the frequency of their histogram bin
  inside the cell; growth from each unassigned seed while |V - V_seed| <= T_int and the
  angular difference of Sobel orientation to the seed <= T_ori, 8-connected, confined to the cell.
* **88 GLCM features**: 22 Haralick/Soh/Clausi features x 4 angles, 16 levels, distance 1,
  computed on pixels inside each region only.
* **PARAMS of D0** (T_int = 0.05, T_ori = pi/4, 256 bins): placeholders.
* **Purity**: majority-class pixels per region over all labelled pixels considered.
* **Search budget** (mu = 4, lambda = 4, G = 10) and the LLM (model, temperature): placeholders.
* **Choice of LLM-MRG**: highest search fitness among designs whose trace shows `B.mrg` with
  grid 3 and no other region-forming block.
* **MRG-PSO / MRG-ASVM**: binary PSO and binary GOA wrappers over the 88 + B features,
  fitness 0.99 (1 - CV acc) + 0.01 (fraction selected).
* **DoE**: Latin hypercube over D0's numbers with the same evaluation budget as the search.
* **Spatially disjoint check**: 16 x 16 tiles assigned whole to train or test.
* **Statistical tests**: McNemar per scene and split, Wilcoxon and Friedman across (scene, split).
