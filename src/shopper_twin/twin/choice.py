"""What a shopper looks at and what they pick on a trip: a discrete choice model with per-shopper parameters.

Two parts share each shopper's taste vector and each product's embedding:

  Views (which products catch their eye), an ordered top-k draw from the whole category:
      s_ui = view_bias_i + view_taste * (taste_u . emb_i) + view_promo * promo_i + view_habit * [bought last time]

  Choice among the viewed products, or buy nothing (the classic multinomial logit):
      V_ui = bias_i + taste_u . emb_i - price_sens_u * log(price_i / category median)
             + store_u * [store brand] + loyalty_u * brand share_ui + habit_u * [bought last time] + promo * promo_i
      V_nothing = outside_c + outside_u

Every per-shopper number = the average for their group (age band x household size, from the public
profile) + their own deviation, and the deviations are penalised. Shoppers with little history stay close
to their group ("pooling"); shoppers with lots of history get their own values.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .trips import MAX_VIEWS, Catalogue, Trips

SHOPPER_TRAITS = ("price_sens", "store", "loyalty", "habit", "outside")


class ChoiceModel(nn.Module):
    def __init__(self, n_shoppers: int, n_products: int, n_categories: int, groups: np.ndarray, dim: int = 16,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.register_buffer("group", torch.as_tensor(groups, dtype=torch.long))
        n_groups = int(groups.max()) + 1

        def zeros(*shape):
            return nn.Parameter(torch.zeros(*shape))

        def small(*shape):
            return nn.Parameter(0.1 * torch.randn(*shape, generator=g))

        # products
        self.bias = zeros(n_products)
        self.view_bias = zeros(n_products)
        self.emb = small(n_products, dim)
        # shoppers: group averages + own deviations
        self.taste_group = zeros(n_groups, dim)
        self.taste_dev = small(n_shoppers, dim)
        self.trait_group = nn.Parameter(torch.tensor([[1.0, 0.0, 0.5, 1.0, 0.0]]).repeat(n_groups, 1))
        self.trait_dev = zeros(n_shoppers, len(SHOPPER_TRAITS))
        # everyone
        self.outside_cat = zeros(n_categories)
        self.promo = nn.Parameter(torch.tensor(0.2))
        self.view_taste = nn.Parameter(torch.tensor(0.5))
        self.view_promo = nn.Parameter(torch.tensor(0.5))
        self.view_habit = nn.Parameter(torch.tensor(1.0))

    # ---- per-shopper values
    def taste(self, u: torch.Tensor) -> torch.Tensor:
        return self.taste_group[self.group[u]] + self.taste_dev[u]

    def traits(self, u: torch.Tensor) -> torch.Tensor:
        """[len(u), 5] price sensitivity, store-brand liking, brand loyalty, habit, outside-option offset."""
        return self.trait_group[self.group[u]] + self.trait_dev[u]

    def penalty(self, taste_weight: float, trait_weight: float, emb_weight: float) -> torch.Tensor:
        return (taste_weight * self.taste_dev.pow(2).sum() + trait_weight * self.trait_dev.pow(2).sum()
                + emb_weight * self.emb.pow(2).sum())

    # ---- likelihoods on a batch of trips
    def choice_logits(self, u, c, viewed, lp, promo, store, share, habit):
        """[B, MAX_VIEWS + 1] utilities of each viewed product and (last column) of buying nothing."""
        tr = self.traits(u)
        safe = viewed.clamp(min=0)
        v = (self.bias[safe] + torch.einsum("bd,bkd->bk", self.taste(u), self.emb[safe])
             - tr[:, 0:1] * lp + tr[:, 1:2] * store + tr[:, 2:3] * share + tr[:, 3:4] * habit + self.promo * promo)
        v = v.masked_fill(viewed < 0, float("-inf"))
        outside = self.outside_cat[c] + tr[:, 4]
        return torch.cat([v, outside[:, None]], dim=1)

    def view_scores(self, u, members, promo, habit):
        """[B, M] how likely each product of ONE category is to be viewed. members [M] (no padding)."""
        taste_fit = self.taste(u) @ self.emb[members].T
        return self.view_bias[members][None, :] + self.view_taste * taste_fit + self.view_promo * promo \
            + self.view_habit * habit


def ordered_view_loglik(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Log-probability of viewing products in the given order (Plackett-Luce top-k).

    scores [B, M]; positions [B, K] index into M, -1 for unused slots. Returns [B].
    Each view is a softmax over the products not viewed yet; the shrinking denominator is
    computed from the running share of probability already used up.
    """
    used = positions >= 0
    log_share = scores.gather(1, positions.clamp(min=0)) - torch.logsumexp(scores, dim=1, keepdim=True)
    share = torch.where(used, log_share.exp(), torch.zeros_like(log_share))
    remaining = (1.0 - (torch.cumsum(share, dim=1) - share)).clamp(min=1e-6)
    term = log_share - remaining.log()
    return torch.where(used, term, torch.zeros_like(term)).sum(dim=1)


class TripTensors:
    """Everything the model needs per trip, precomputed once as tensors."""

    def __init__(self, trips: Trips, cat: Catalogue):
        lp_all = cat.log_price_ratio()
        week = np.minimum(trips.week, cat.n_weeks - 1)
        viewed = trips.viewed
        safe = np.maximum(viewed, 0)
        self.u = torch.as_tensor(trips.shopper)
        self.c = torch.as_tensor(trips.category)
        self.viewed = torch.as_tensor(viewed)
        self.target = torch.as_tensor(np.where(trips.chosen >= 0, trips.chosen, MAX_VIEWS))
        self.lp = torch.as_tensor(np.where(viewed >= 0, lp_all[safe, week[:, None]], 0).astype(np.float32))
        self.promo = torch.as_tensor(np.where(viewed >= 0, cat.promo[safe, week[:, None]], 0).astype(np.float32))
        self.store = torch.as_tensor(np.where(viewed >= 0, cat.is_store[safe], 0).astype(np.float32))
        self.share = torch.as_tensor(trips.brand_share)
        self.habit = torch.as_tensor(((viewed == trips.last_bought[:, None]) & (viewed >= 0)).astype(np.float32))
        # views: position of each viewed product inside its category's member list
        slot_in_cat = np.zeros(cat.n_products, dtype=np.int64)
        m = cat.members
        r, k = np.nonzero(m >= 0)
        slot_in_cat[m[r, k]] = k
        self.positions = torch.as_tensor(np.where(viewed >= 0, slot_in_cat[safe], -1))
        self.week = torch.as_tensor(week)
        self.last_bought = torch.as_tensor(trips.last_bought)
        self.weight = torch.ones(len(trips))

    def __len__(self) -> int:
        return len(self.u)


def _category_batches(data: TripTensors, n_categories: int, batch_size: int, rng) -> list[np.ndarray]:
    """Trip indices in batches that each hold one category, so the view model only scores that category."""
    c = data.c.numpy()
    by_cat = [np.flatnonzero(c == k) for k in range(n_categories)]
    order = (lambda rows: rng.permutation(rows)) if rng is not None else (lambda rows: rows)
    return [chunk for rows in by_cat if len(rows)
            for chunk in np.array_split(order(rows), -(-len(rows) // batch_size))]


def _batch_loglik(model: ChoiceModel, data: TripTensors, idx: torch.Tensor, members: torch.Tensor,
                  promo_all: torch.Tensor, sizes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-trip log-likelihood of the choice made and of the views, for trips of one category."""
    c0 = int(data.c[idx[0]])
    u, c = data.u[idx], data.c[idx]
    logits = model.choice_logits(u, c, data.viewed[idx], data.lp[idx], data.promo[idx], data.store[idx],
                                 data.share[idx], data.habit[idx])
    choice_ll = -nn.functional.cross_entropy(logits, data.target[idx], reduction="none")
    mem = members[c0, :sizes[c0]]
    vpromo = promo_all[mem][:, data.week[idx]].T
    vhabit = (mem[None, :] == data.last_bought[idx][:, None]).float()
    view_ll = ordered_view_loglik(model.view_scores(u, mem, vpromo, vhabit), data.positions[idx])
    return choice_ll, view_ll


def fit_choice_model(trips: Trips, cat: Catalogue, groups: np.ndarray, n_shoppers: int, dim: int = 16,
                     epochs: int = 6, batch_size: int = 4096, lr: float = 0.03, view_weight: float = 1.0,
                     taste_reg: float = 1.0, trait_reg: float = 2.0, emb_reg: float = 0.1,
                     trip_weights: np.ndarray | None = None, seed: int = 0, verbose: bool = False) -> ChoiceModel:
    """Maximum a-posteriori fit with Adam on minibatches of trips (trips with weight 0 are left out)."""
    torch.manual_seed(seed)
    data = TripTensors(trips, cat)
    if trip_weights is not None:
        data.weight = torch.as_tensor(trip_weights, dtype=torch.float32)
    model = ChoiceModel(n_shoppers, cat.n_products, cat.n_categories, groups, dim=dim, seed=seed)
    members = torch.as_tensor(cat.members)
    promo_all = torch.as_tensor(cat.promo)
    sizes = (cat.members >= 0).sum(axis=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total_weight = float(data.weight.sum())
    keep = data.weight.numpy() > 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        t0, running = time.perf_counter(), 0.0
        batches = [b[keep[b]] for b in _category_batches(data, cat.n_categories, batch_size, rng)]
        batches = [b for b in batches if len(b)]
        for b_i in rng.permutation(len(batches)):
            idx = torch.as_tensor(batches[b_i])
            w = data.weight[idx]
            choice_ll, view_ll = _batch_loglik(model, data, idx, members, promo_all, sizes)
            # minibatch estimate of (mean negative log-likelihood + prior penalty / number of trips)
            loss = (-(w * (choice_ll + view_weight * view_ll)).sum() / w.sum()
                    + model.penalty(taste_reg, trait_reg, emb_reg) / total_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(idx)
        if verbose:
            print(f"    choice epoch {epoch + 1}/{epochs}: loss {running / keep.sum():.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")
    return model.eval().requires_grad_(False)


def trip_loglik(model: ChoiceModel, trips: Trips, cat: Catalogue, batch_size: int = 8192) -> dict[str, float]:
    """Average log-likelihood per trip of what was chosen and of what was viewed (higher is better).
    Used to compare settings on held-out trips: it checks the per-shopper numbers, not just the ranking."""
    data = TripTensors(trips, cat)
    members = torch.as_tensor(cat.members)
    promo_all = torch.as_tensor(cat.promo)
    sizes = (cat.members >= 0).sum(axis=1)
    choice_sum = view_sum = 0.0
    with torch.no_grad():
        for b in _category_batches(data, cat.n_categories, batch_size, None):
            choice_ll, view_ll = _batch_loglik(model, data, torch.as_tensor(b), members, promo_all, sizes)
            choice_sum += float(choice_ll.double().sum())
            view_sum += float(view_ll.double().sum())
    n = max(len(data), 1)
    return {"choice_loglik": choice_sum / n, "view_loglik": view_sum / n, "trips": len(data)}
