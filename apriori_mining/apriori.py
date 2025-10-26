from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from .data import FastContext
try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(x, **kwargs):  # type: ignore
        return x


def _support_counts(transactions: List[Set[str]], candidates: Iterable[frozenset]) -> Dict[frozenset, int]:
    counts: Dict[frozenset, int] = defaultdict(int)
    for tx in transactions:
        for c in candidates:
            if c.issubset(tx):
                counts[c] += 1
    return counts


def _support_counts_fast(candidates: Iterable[frozenset], ctx: FastContext, progress: bool = False) -> Dict[frozenset, int]:
    if ctx.masks is None:
        raise RuntimeError("Fast context masks missing")
    masks = np.asarray(ctx.masks)
    counts: Dict[frozenset, int] = {}
    iterable = list(candidates)
    for c in _tqdm(iterable, disable=not progress, desc="Support count", leave=False):
        combo_mask = 0
        for item in c:
            combo_mask |= (1 << ctx.name_to_idx[item])
        cnt = int(np.count_nonzero((masks & combo_mask) == combo_mask))
        counts[c] = cnt
    return counts


def _generate_candidates(prev_frequents: List[frozenset], k: int) -> Set[frozenset]:
    candidates: Set[frozenset] = set()
    prev_frequents_sorted = [sorted(list(s)) for s in prev_frequents]
    prev_set = set(prev_frequents)
    n = len(prev_frequents_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            a = prev_frequents_sorted[i]
            b = prev_frequents_sorted[j]
            if a[: k - 2] == b[: k - 2]:
                candidate = frozenset(set(a) | set(b))
                if len(candidate) == k:
                    all_subsets_frequent = True
                    for subset in combinations(candidate, k - 1):
                        if frozenset(subset) not in prev_set:
                            all_subsets_frequent = False
                            break
                    if all_subsets_frequent:
                        candidates.add(candidate)
    return candidates


def apriori(
    transactions: List[Set[str]],
    min_support: float,
    ctx: FastContext,
    progress: bool = False,
) -> Tuple[Dict[int, Dict[frozenset, float]], Dict[frozenset, int]]:
    num_tx = len(transactions)
    items = sorted({i for tx in transactions for i in tx})
    c1 = [frozenset([i]) for i in items]
    if ctx.matrix is not None:
        matrix = np.asarray(ctx.matrix, dtype=bool)
        c1_counts: Dict[frozenset, int] = {}
        col_sums = np.asarray(matrix, dtype=np.int32).sum(axis=0)
        for item in items:
            idx = ctx.name_to_idx[item]
            c1_counts[frozenset([item])] = int(col_sums[idx])
    else:
        c1_counts = _support_counts(transactions, c1)

    l1 = {s: cnt / num_tx for s, cnt in c1_counts.items() if (cnt / num_tx) >= min_support}
    frequents_by_k: Dict[int, Dict[frozenset, float]] = {1: dict(sorted(l1.items(), key=lambda x: (-x[1], sorted(list(x[0])))))}
    all_counts: Dict[frozenset, int] = {s: c for s, c in c1_counts.items()}

    k = 2
    prev_frequents = list(l1.keys())
    while prev_frequents:
        candidates = _generate_candidates(prev_frequents, k)
        if not candidates:
            break
        if ctx.masks is not None:
            counts_k = _support_counts_fast(candidates, ctx, progress=progress)
        else:
            counts_k = _support_counts(transactions, candidates)
        all_counts.update(counts_k)
        lk = {s: cnt / num_tx for s, cnt in counts_k.items() if (cnt / num_tx) >= min_support}
        if not lk:
            break
        frequents_by_k[k] = dict(sorted(lk.items(), key=lambda x: (-x[1], sorted(list(x[0])))))
        prev_frequents = list(lk.keys())
        k += 1

    return frequents_by_k, all_counts
