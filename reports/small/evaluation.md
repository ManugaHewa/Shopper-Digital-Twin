## How right is the twin? Scored against the true store (days 152-179)

29 what-if scenarios were run through the true store (3 runs each, with the answer key) and through each method. A miss is the gap in percentage points between the predicted and the true % change: if the truth is -5.3% and the twin says -6.1%, the miss is 0.8 points.

On the changed products' units the twin misses by 8.1 points on average, against 14.3 for the store-wide elasticity model.

### Average miss (percentage points, lower is better)

| Method | Changed products' units | Whole category's units | Category revenue | Typical miss vs the true change (changed products) |
|---|---|---|---|---|
| Twin | 8.1 | 0.7 | 0.8 | 14% |
| Twin without personal traits | 10.6 | 0.7 | 0.8 | 15% |
| Store-wide elasticity model | 14.3 | 1.5 | 1.5 | 17% |
| No reaction | 82.3 | 4.2 | 4.3 | 100% |

Changed products' units are scored on the 22 price and promotion scenarios (a stock-out always takes them to -100%, which every method gets right). "Typical miss vs the true change" is the median of miss / size of the true change: 10% means an answer of about -9% or -11% when the truth is -10%. The true answer itself has a small margin of error from the random runs (median standard error 0.72 points).

### The five standard questions

Changed products' units (for the stock-out: the whole category's units), true vs predicted:

| Scenario | Truth | Twin | Without personal traits | Store-wide elasticity | No reaction |
|---|---|---|---|---|---|
| Coffee +20% | -5.3% | -6.1% | -6.1% | -6.5% | +0.0% |
| Store-brand milk -15% | +16.3% | +12.6% | +12.4% | +15.4% | +0.0% |
| Best-selling cereal out of stock (Riverridge Breakfast Cereal Pro) | -0.8% | -0.7% | -1.2% | -0.0% | -7.7% |
| 25% off Rivermark snacks | +175.9% | +173.4% | +168.8% | +169.6% | +0.0% |
| All store-brand products +10% | -7.3% | -6.0% | -6.1% | -8.0% | +0.0% |

### Is the twin's range honest?

The 90% bands in the what-if report come from resampling shoppers, so they only show how much the answer depends on who is in the store. Against the truth:

| | Band contains the truth | 90% of misses are under | ... or as a share of the true change |
|---|---|---|---|
| Changed products' units | 9% of 22 | 22.0 points | 22% |
| Whole category's units | 10% of 29 | 1.4 points | - |
| Category revenue | 10% of 29 | 1.9 points | - |

So quote a twin answer with the error it really has (the last two columns), not with the narrow band.

### Does it know which shoppers react?

The shoppers split into four equal groups by their TRUE price sensitivity (from the answer key). Over the 22 price and promotion scenarios:

| | Average miss per group (points) | Gap between the most and least sensitive group |
|---|---|---|
| Truth | | 78.0 points |
| Twin | 14.7 | 46.9 points |
| Twin without personal traits | 22.6 | 18.8 points |

Coffee +20%, change in the changed products' units by true price sensitivity:

| | lowest | low | high | highest |
|---|---|---|---|---|
| Truth | -3.1% | -6.3% | -5.7% | -5.6% |
| Twin | -5.5% | -6.3% | -6.4% | -6.0% |
| Without personal traits | -7.3% | -6.6% | -6.0% | -5.3% |

### Did it learn each shopper's traits?

Rank correlation between learned and true values over all shoppers (1 = same order, 0 = no relation; 90% range in brackets). "Profile only" is what the public profile (age band and household size) gives on its own.

| Trait | Twin | Profile only | Least history | Most history | Median learned vs true |
|---|---|---|---|---|---|
| price sensitivity | 0.79 (0.78-0.79) | 0.30 | 0.63 | 0.86 | 1.62 vs 1.55 |
| store-brand liking | 0.84 (0.83-0.84) | 0.27 | 0.72 | 0.85 | different scale |
| brand loyalty | 0.13 (0.11-0.15) | 0.01 | 0.09 | 0.19 | different scale |
| habit | 0.55 (0.54-0.56) | 0.01 | 0.41 | 0.67 | 1.01 vs 0.93 |

Taste: within each category, the twin orders products by appeal with an average rank correlation of 0.64 with each shopper's true order (1,000 shoppers sampled), against 0.40 for the same popularity order for everyone.

### Are its purchase chances right?

Every shopper x product (50,000,000 pairs), chance of buying in the window vs what happened (146,073 pairs bought):

| Forecast | Pairs bought (predicted) | Brier score | Log loss | Better than own history by |
|---|---|---|---|---|
| Twin | 146,287 | 0.002594 | 0.01302 | +6.4% (Brier), +18.2% (log loss) |
| Popularity | 147,387 | 0.002894 | 0.01795 | -4.4% (Brier), -12.8% (log loss) |
| Own history | 153,676 | 0.002772 | 0.01592 | +0.0% (Brier), +0.0% (log loss) |

Twin, by predicted chance:

| Predicted chance | Pairs | Average predicted | Actually bought |
|---|---|---|---|
| 0.0%-0.2% | 42,347,449 | 0.02% | 0.03% |
| 0.2%-0.5% | 3,186,096 | 0.32% | 0.39% |
| 0.5%-1.0% | 1,809,516 | 0.71% | 0.79% |
| 1.0%-2.0% | 1,276,953 | 1.41% | 1.43% |
| 2.0%-5.0% | 915,094 | 3.06% | 2.78% |
| 5.0%-10.0% | 271,498 | 6.83% | 5.91% |
| 10.0%-20.0% | 103,476 | 13.72% | 12.84% |
| 20.0%-40.0% | 55,876 | 28.41% | 27.32% |
| 40.0%-60.0% | 23,756 | 48.57% | 43.58% |
| 60.0%-100.0% | 10,286 | 70.40% | 60.02% |

### Where it misses most (whole category's units)

- Pet Toys +20%: truth -15.6%, twin -12.8%
- Everyday Value Baby Formula Plus out of stock: truth -1.7%, twin +0.6%
- Grills & Outdoor -10%: truth +9.3%, twin +7.7%

Speed: the twin answers a scenario in 2s for 10,000 shoppers; one run of the true store takes 7s (and only exists because the store is simulated).
