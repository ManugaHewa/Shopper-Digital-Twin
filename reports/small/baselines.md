### All purchases (9,569 shoppers, next 28 days)

| Model | NDCG@10 | Hit rate@10 | Recall@10 | Precision@10 | Coverage@10 | Fit (s) | ms/shopper |
|---|---|---|---|---|---|---|---|
| Blend (cycle + ALS) | 0.352 | 0.924 | 0.218 | 0.294 | 0.519 | 27.4 | 0.768 |
| Repurchase cycle | 0.328 | 0.919 | 0.211 | 0.278 | 0.871 | 0.5 | 0.129 |
| Buy again | 0.284 | 0.883 | 0.175 | 0.235 | 0.889 | 0.1 | 0.128 |
| Matrix factorisation (ALS) | 0.201 | 0.835 | 0.140 | 0.177 | 0.326 | 26.9 | 0.111 |
| Item-to-item | 0.088 | 0.509 | 0.053 | 0.080 | 0.102 | 1.1 | 0.129 |
| Popularity | 0.055 | 0.328 | 0.029 | 0.051 | 0.002 | 0.0 | 0.218 |
| Random | 0.003 | 0.030 | 0.002 | 0.003 | 1.000 | 0.0 | 0.155 |

### New-to-shopper products (9,519 shoppers, next 28 days)

| Model | NDCG@10 | Hit rate@10 | Recall@10 | Precision@10 | Coverage@10 | Fit (s) | ms/shopper |
|---|---|---|---|---|---|---|---|
| Matrix factorisation (ALS) | 0.067 | 0.407 | 0.060 | 0.053 | 0.134 | 11.3 | 0.105 |
| Blend (cycle + ALS) | 0.067 | 0.407 | 0.060 | 0.053 | 0.134 | 11.8 | 0.748 |
| Item-to-item | 0.045 | 0.300 | 0.040 | 0.037 | 0.230 | 0.9 | 0.117 |
| Popularity | 0.017 | 0.132 | 0.014 | 0.015 | 0.006 | 0.1 | 0.224 |
| Buy again | 0.017 | 0.132 | 0.014 | 0.015 | 0.006 | 0.1 | 0.135 |
| Repurchase cycle | 0.017 | 0.132 | 0.014 | 0.015 | 0.006 | 0.5 | 0.137 |
| Random | 0.002 | 0.018 | 0.002 | 0.002 | 1.000 | 0.0 | 0.145 |

Settings tuned on the validation window; scored once on the test window.
