# Shopper Digital Twin: Plan

**One-line pitch:** every customer gets a simulated "twin" learned from their past browsing and buying. Before changing a price, launching a product or removing one, you run the change on the twins and read off what the real customers would most likely do.

**Language:** Python throughout (pandas/Polars, NumPy, PyTorch, scikit-learn, FAISS, Streamlit for the demo). This is a default; say if you'd rather use something else.

---

## 1. What building it involves (the simple version)

| # | Part | What it is | Output |
|---|------|------------|--------|
| 1 | **Fake world** | A generator that invents products, shoppers and months of sessions. Each fake shopper has *hidden* true tastes and price sensitivity. | Event logs (views, carts, buys) + a sealed "answer key" |
| 2 | **Features** | Turn raw logs into what the model needs: product embeddings, per-shopper history, prices seen, time/context. | Clean tables |
| 3 | **Twin model** | Learns each shopper's tastes, price sensitivity and habits from their logs only (it never sees the answer key). | One parameter set per shopper |
| 4 | **Simulator** | Runs the twins forward in time: who shows up, what they browse, what they buy. | Simulated sales under any scenario |
| 5 | **What-if engine** | Change the world (price -10%, new product, stock-out), re-simulate, compare to baseline. | Demand change with uncertainty bands |
| 6 | **Evaluation** | Compare twin predictions to the answer key and to held-out real behaviour. | Accuracy report |
| 7 | **Demo** | Interactive app: pick a product, drag the price slider, watch predicted demand and individual twins react. | Portfolio showpiece |

The fake world with a hidden answer key is the project's strongest point. Real companies can never check a what-if prediction because the alternate world never happens. Here you can, because you built the world and can re-run it with the change applied.

---

## 2. How the algorithm works

### 2.1 The fake world (ground truth)

- **Products (~5k):** category, brand, price, quality, and a hidden "style vector" (e.g. 16 numbers describing what the product is like).
- **Shoppers (~50k to 100k):** a hidden taste vector in the same space, a price sensitivity, a brand loyalty, a shopping frequency, a budget, and life-cycle traits (e.g. buys groceries weekly, electronics yearly).
- **Behaviour rules:** each day a shopper may visit (probability from frequency). On a visit they browse items that roughly match their taste, then buy with a probability from the utility formula below, or leave with nothing.
- **Noise and realism:** seasonality, promotions, occasional random clicks, taste drift over time, a few "life events" that shift preferences.
- Store the hidden parameters separately. The model never reads them; only evaluation does.

### 2.2 The twin: a choice model per shopper

At the heart of every twin is one idea from economics (the discrete choice / multinomial logit model): when a shopper looks at a set of options, they pick each one with probability proportional to how much they like it.

For shopper *u* and product *i*, the twin estimates a **utility score**:

> utility = taste match + habit + context − price sensitivity × price + noise

- **Taste match:** dot product of the shopper's learned taste vector and the product's embedding.
- **Habit:** bonus for brands/categories they buy repeatedly and for items due for repurchase (e.g. 30 days since last coffee).
- **Context:** time of day, season, whether it's on promotion.
- **Price term:** each shopper has their own price sensitivity. This is the number that makes "what if the price changes" answerable.

The probability of buying *i* out of the options shown is a **softmax** over utilities, including a "buy nothing" option so the model can predict lost sales, not only switching.

### 2.3 Learning the twins (training)

1. **Product embeddings:** learn from co-viewing and co-buying (items browsed together sit close together), e.g. matrix factorisation or item2vec.
2. **Shopper parameters:** fit taste vector, price sensitivity and habit weights by maximising the likelihood of the choices each shopper actually made.
3. **Hierarchical pooling:** a shopper with 3 purchases can't be learned alone. Each shopper's parameters are pulled towards the average of similar shoppers, and pulled less as their own data grows. This is the standard "hierarchical Bayesian" trick, done with PyTorch (MAP fit) or a variational approximation.
4. **Sequence layer (optional upgrade):** a small transformer (SASRec-style) over each shopper's recent clicks captures "what are they in the middle of right now". Its output feeds the context term.
5. **Uncertainty:** keep a spread, not just a point estimate, for each shopper's parameters so predictions come with confidence bands.

### 2.4 Running a twin (simulation loop)

For each simulated day, for every twin, vectorised across all twins at once:

1. **Arrive?** Draw from the twin's visit rate (a Poisson process with seasonality).
2. **Browse:** retrieve the top ~50 candidate products for this twin with fast nearest-neighbour search (FAISS) on taste vs. product embeddings. This mirrors how real shops show a limited set of items.
3. **Choose:** compute utilities for those candidates plus "buy nothing", softmax, sample a choice.
4. **Update state:** record the purchase, reset repurchase clocks, spend budget, let taste drift a little.

Repeat for 30 to 90 simulated days and total up sales, revenue and who bought what. Run it several times with different parameter draws to get uncertainty bands.

### 2.5 Asking what-if questions

A scenario is simply a change to the world the twins see:

- **Price change:** edit the price; the price term shifts every twin's utility.
- **New product:** give it an embedding (from its attributes) and insert it into the candidate pool.
- **Remove / stock-out:** delete it from the pool and see where demand moves.
- **Promotion:** add the promo bonus for chosen days.

Run baseline and scenario with the **same random seeds** so differences come from the change, not from luck. Report demand change, revenue change, who switched to what (cannibalisation), and which shopper segments react most.

### 2.6 The AI-agent layer (the modern twist)

The choice model does the numbers; an LLM gives each twin a voice.

- Build a short persona per twin from its learned parameters ("price-sensitive, loyal to Brand A, weekly grocery buyer, drifting towards organic").
- In the demo, a user can "interview" a twin: *"Why didn't you buy the headphones at £79?"* The LLM answers using that twin's actual utility breakdown, so the explanation matches the maths.
- Optional experiment: let the LLM agent make choices directly and compare its accuracy with the choice model. That comparison is itself an interesting finding for the portfolio.

The LLM never overrides the numbers used for predictions; this keeps results reproducible and cheap to run on 100k shoppers.

---

## 3. How we'll know it works

| Test | Question it answers | Metric |
|------|---------------------|--------|
| Next-purchase prediction | Does the twin know what this shopper buys next? | Hit rate@10, NDCG@10 on held-out weeks |
| Parameter recovery | Did we learn the hidden tastes and price sensitivities? | Correlation of learned vs. true values |
| What-if accuracy | If we really change the price in the fake world, does the twin predict the result? | Error in predicted vs. actual demand change |
| Calibration | When the twin says 30% chance, is it right 30% of the time? | Calibration curve, Brier score |
| Baselines | Is it better than simple approaches? | vs. "most popular", plain collaborative filtering, one elasticity for everyone |
| Speed | Big-tech-style efficiency | Twins simulated per second, candidate retrieval latency |

The what-if accuracy test is the headline: re-run the fake world's true rules with the change applied, then check how close the twins came.

---

## 4. Build order (milestones)

1. **Fake world v1:** products, shoppers, sessions, answer key. Sanity plots.
2. **Baselines:** popularity and matrix factorisation recommenders, so there's something to beat.
3. **Twin v1:** per-shopper choice model with price sensitivity and pooling. Next-purchase metrics.
4. **Simulator + what-if engine:** price and stock-out scenarios with uncertainty bands.
5. **Evaluation report:** parameter recovery and what-if accuracy against the answer key.
6. **Upgrades:** sequence transformer, taste drift, life events.
7. **Demo app:** price slider, segment breakdown, "interview a twin" chat.
8. **Write-up:** README with results charts, ready for a portfolio.

## 5. Scale targets

- 100k shoppers, 5k products, ~10M events generated in minutes.
- Training on a laptop CPU/GPU in under an hour.
- A 30-day what-if simulation for all twins in seconds, thanks to vectorised maths and FAISS retrieval.

## 6. Decisions still open

- **Domain:** decided 2026-09-25: general Walmart-style store (covers both repeat buying and taste).
- **LLM for twin interviews:** Claude via API (default), or skip the chat and keep it pure ML.
- **Repository:** none attached yet; code needs a GitHub repo to live in.
