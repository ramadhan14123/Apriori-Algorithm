from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass
class FastContext:
    matrix: np.ndarray | None
    masks: np.ndarray | None
    name_to_idx: Dict[str, int]
    item_names: List[str]
    progress: bool = False


def _dedupe_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 1
            result.append(h)
    return result


def read_transactions(csv_path: str, progress: bool = False) -> Tuple[List[Set[str]], List[str], FastContext]:
    transactions: List[Set[str]] = []
    if pd is not None:
        df = pd.read_csv(csv_path)
        headers = _dedupe_headers([str(h).strip() for h in df.columns.tolist()])
        df.columns = headers
        values = (df.values.astype(np.int8) > 0)
        item_names = headers
        name_to_idx = {name: i for i, name in enumerate(item_names)}
        for row in values:
            idxs = np.nonzero(row)[0]
            transactions.append({item_names[i] for i in idxs})
        masks = np.zeros(values.shape[0], dtype=np.uint32)
        for r_idx, row in enumerate(values):
            mask = 0
            for i in np.nonzero(row)[0]:
                mask |= (1 << i)
            masks[r_idx] = mask
        ctx = FastContext(matrix=values, masks=masks, name_to_idx=name_to_idx, item_names=item_names, progress=progress)
        return transactions, item_names, ctx
    else:
        # Fallback CSV
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            raw_headers = next(reader)
            headers = _dedupe_headers([h.strip() for h in raw_headers])
            item_names = headers
            for row in reader:
                vals = [int(v.strip()) if v.strip() != "" else 0 for v in row]
                tx_items = {name for name, v in zip(headers, vals) if v == 1}
                transactions.append(tx_items)
        ctx = build_fast_context(transactions, item_names, progress=progress)
        return transactions, item_names, ctx


def build_fast_context(transactions: List[Set[str]], item_names: Sequence[str], progress: bool = False) -> FastContext:
    n = len(item_names)
    matrix = np.zeros((len(transactions), n), dtype=bool)
    name_to_idx = {name: i for i, name in enumerate(item_names)}
    for r, tx in enumerate(transactions):
        for item in tx:
            matrix[r, name_to_idx[item]] = True
    masks = np.zeros(matrix.shape[0], dtype=np.uint32)
    for r_idx, row in enumerate(matrix):
        mask = 0
        for i in np.nonzero(row)[0]:
            mask |= (1 << i)
        masks[r_idx] = mask
    return FastContext(matrix=matrix, masks=masks, name_to_idx=name_to_idx, item_names=list(item_names), progress=progress)
