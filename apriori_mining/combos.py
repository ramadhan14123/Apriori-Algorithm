from __future__ import annotations
from itertools import combinations
from math import comb
from typing import List, Sequence, Set, Tuple

import numpy as np
try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(x, **kwargs):  # type: ignore
        return x

from .data import FastContext


def all_combinations_support(
    transactions: List[Set[str]],
    item_names: Sequence[str],
    max_k: int | None,
    ctx: FastContext,
) -> List[Tuple[Tuple[str, ...], int, float]]:
    num_tx = len(transactions)
    if ctx.masks is not None:
        tx_masks = np.asarray(ctx.masks, dtype=np.uint32)
    else:
        name_to_idx = {name: idx for idx, name in enumerate(item_names)}
        tx_masks = np.zeros(num_tx, dtype=np.uint32)
        for r, tx in enumerate(transactions):
            mask = 0
            for item in tx:
                mask |= (1 << name_to_idx[item])
            tx_masks[r] = mask

    n = len(item_names)
    if max_k is None:
        max_k = n

    results: List[Tuple[Tuple[str, ...], int, float]] = []
    for k in range(1, max_k + 1):
        iterator = combinations(range(n), k)
        total = comb(n, k)
        for idxs in _tqdm(iterator, total=total, desc=f"All combos k={k}", leave=False, disable=not ctx.progress):
            combo_mask = 0
            for idx in idxs:
                combo_mask |= 1 << idx
            count = int(np.count_nonzero((tx_masks & combo_mask) == combo_mask))
            if count > 0:
                items = tuple(item_names[i] for i in idxs)
                results.append((items, count, count / num_tx))

    results.sort(key=lambda r: (-r[2], r[0]))
    return results
