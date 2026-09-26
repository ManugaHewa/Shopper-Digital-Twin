# Shopper Digital Twin

Predict what every customer would buy **before** a change happens.

Each shopper in a simulated Walmart-style general store gets a "twin", learned only from their browsing and purchase history. The twins are run forward in a simulator to answer what-if questions: *What happens to demand if this price goes up 10%? Where do shoppers go if this product is out of stock? Who buys the new product?*

Because the store and its shoppers are generated, every shopper's true tastes and price sensitivity are known. That makes it possible to check the twins' what-if forecasts against the real answer, something retailers can never do.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
pip install -e ".[dev]"

python scripts/generate_world.py --preset small     # 10k shoppers, 180 days, ~10M events (~30 s)
python scripts/inspect_world.py --world data/worlds/small
python scripts/run_baselines.py --world data/worlds/small   # tune and score the baselines (~10 min)
pytest
```

Presets: `tiny` (1k shoppers, 60 days), `small` (10k shoppers, 180 days), `full` (100k shoppers, 180 days, ~100M events).

## How it works

- **Synthetic world** (`src/shopper_twin/world/`): a general store with 81 categories across 13 departments, ~5k products, and shoppers from six hidden segments. Every shopper has a sealed answer key of true parameters: taste, price sensitivity, quality preference, brand loyalty, habits, pets, kids, repurchase cycles.
- **Twin model:** a per-shopper discrete choice model
  `utility = taste match + habit + context - price sensitivity x price`
  with a "buy nothing" option and hierarchical pooling for shoppers with little history.
- **Simulator:** day-by-day, vectorised over all twins, with FAISS candidate retrieval.
- **What-if engine:** re-simulate with the same random seeds and compare against the baseline.
- **Evaluation:** next-purchase accuracy against strong baselines, parameter recovery, what-if accuracy, calibration, speed.
- **Demo:** Streamlit app with a price slider and a chat to interview any twin.

## Does the fake store behave like a real one?

From `scripts/inspect_world.py` on the `small` preset (10k shoppers, 180 days):

| Check | Result |
|---|---|
| Average basket | 3.7 items, $49.50 |
| Grocery share of purchases | 66% |
| Weekend vs. weekday revenue | 1.48x |
| Staple repurchase timing vs. design | log correlation 0.997 |
| Store-brand share, least vs. most price-sensitive shoppers | 15% vs. 57% |
| Coffee +40% price (what-if test, 2k shoppers, 3 seeds) | 11-16% fewer units sold |

## How good are the baselines?

The twin has to beat these. From `scripts/run_baselines.py` on the `small` preset: each model is tuned on days 124-151, then scored once on days 152-179 (9,569 shoppers). The score is NDCG@10, where 1.0 means a perfect top-10 list.

| Model | All purchases | New-to-shopper products |
|---|---|---|
| Blend (repurchase cycle + ALS) | **0.351** | **0.067** |
| Repurchase cycle | 0.328 | 0.017 |
| Buy again | 0.284 | 0.017 |
| Matrix factorisation (ALS) | 0.202 | **0.067** |
| Item-to-item | 0.088 | 0.045 |
| Popularity | 0.055 | 0.017 |
| Random | 0.003 | 0.002 |

Most of what people buy in a general store is a repeat, so models built on each shopper's own history lead on all purchases. The repurchase-cycle model, which also knows when each shopper is due to restock, gets at least one of its top 10 right for 92% of shoppers. Products a shopper has never bought are much harder. There ALS leads, with at least one hit for 41% of shoppers, and the tuned blend puts all its weight on it.

![Baseline results](reports/small/baselines.png)

## Project structure

```
src/shopper_twin/world/      fake store generator (spec, catalogue, population, simulator, saving)
src/shopper_twin/data.py     loads a world's public data only (never the answer key)
src/shopper_twin/eval/       time-based splits, ranking metrics, scoring and tuning harness
src/shopper_twin/baselines/  popularity, buy again, repurchase cycle, item-to-item, ALS, blend
scripts/                     command-line entry points
tests/                       pytest checks
data/                        generated worlds (git-ignored)
reports/                     sanity-check charts and baseline results
```

## Status

- [x] Milestone 1: synthetic world generator
- [x] Milestone 2: baseline recommenders and evaluation harness
- [ ] Milestone 3: twin model v1
- [ ] Milestone 4: simulator and what-if engine
- [ ] Milestone 5: evaluation report
- [ ] Milestone 6: upgrades (sequence model, drift, life events)
- [ ] Milestone 7: demo app
- [ ] Milestone 8: write-up

Built in Python.
