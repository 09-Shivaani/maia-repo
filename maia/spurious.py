"""Spurious-feature removal on Spawrious O2O-Easy (paper Sec. 5.1, Table 2, App. F).

Pipeline
  1. train ResNet-18 for one epoch on the (spuriously correlated) train split
     lr 1e-4, weight decay 1e-4, dropout 0.1 on the final layer, 90/10 train/val split (22,810 / 2,534)
  2. extract 512-d final-layer (avg-pooled layer4) features for val / test
  3. Table 2 rows: original model, l1 top-50, random-22, l1 top-22, MILAN / VLM-exemplar labels,
     MAIA-selected units (unregularized logistic regression on the selected units),
     and the balanced-data references (l1 hyper-parameter tuning, l1 top-22 on balanced data).
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from maia.vision.models import IMAGENET_MEAN, IMAGENET_STD

log = logging.getLogger(__name__)

CLASSES = ["bulldog", "corgi", "dachshund", "labrador"]
LOCATIONS = ["desert", "jungle", "dirt", "mountain", "snow", "beach"]
# From the official Spawrious code (aengusl/spawrious): O2O-Easy combinations.
O2O_EASY = {"group": ["desert", "jungle", "dirt", "snow"], "test": ["dirt", "snow", "desert", "jungle"], "filler": "beach"}
_ORDER = ["bulldog", "dachshund", "labrador", "corgi"]  # order used to index group/test lists
TOTAL = 3168


@dataclass
class Sample:
    path: str
    label: int
    location: str


def _list(folder: Path, limit: int | None = None) -> List[str]:
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing - run scripts/download_datasets.py --spawrious")
    files = sorted(str(folder / f) for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg")))
    return files[:limit] if limit else files


def build_o2o_easy(root: str | Path):
    """Return (train_pool, test) sample lists following the official O2O-Easy construction."""
    base = Path(root) / "spawrious224"
    counts = [int(0.97 * TOTAL), int(0.87 * TOTAL)]
    train: List[Sample] = []
    for ind in (0, 1):
        for i, cls in enumerate(_ORDER):
            loc = O2O_EASY["group"][i]
            train += [Sample(p, CLASSES.index(cls), loc) for p in _list(base / f"{ind}/{loc}/{cls}", counts[ind])]
            filler = O2O_EASY["filler"]
            train += [Sample(p, CLASSES.index(cls), filler) for p in _list(base / f"{ind}/{filler}/{cls}", TOTAL - counts[ind])]
    test: List[Sample] = []
    for i, cls in enumerate(_ORDER):
        loc = O2O_EASY["test"][i]
        for ind in (0, 1):
            test += [Sample(p, CLASSES.index(cls), loc) for p in _list(base / f"{ind}/{loc}/{cls}")]
    return train, test


def split_train_val(train_pool: List[Sample], seed: int = 0, frac: float = 0.9):
    rng = random.Random(seed)
    idx = list(range(len(train_pool)))
    rng.shuffle(idx)
    n = int(round(frac * len(idx)))
    return [train_pool[i] for i in idx[:n]], [train_pool[i] for i in idx[n:]]


def split_test_balanced_pool(test: List[Sample], pool_frac: float = 0.2, seed: int = 0):
    """Hold out part of the test-distribution images as a pool for balanced validation sets, so the
    balanced-data baselines never see the images they are evaluated on."""
    rng = random.Random(seed)
    by_group: dict = {}
    for s in test:
        by_group.setdefault((s.label, s.location), []).append(s)
    eval_set, pool = [], []
    for g, items in by_group.items():
        rng.shuffle(items)
        k = int(pool_frac * len(items))
        pool += items[:k]
        eval_set += items[k:]
    return eval_set, pool


def balanced_sets(val: List[Sample], pool: List[Sample], n_sets: int = 10, size: int = 320, seed: int = 0):
    """Balanced validation sets: equal numbers of spuriously-aligned (val) and counter-aligned (pool)
    images for every breed."""
    rng = random.Random(seed)
    per = size // (2 * len(CLASSES))
    sets = []
    for _ in range(n_sets):
        s = []
        for c in range(len(CLASSES)):
            a = [x for x in val if x.label == c]
            b = [x for x in pool if x.label == c]
            s += rng.sample(a, per) + rng.sample(b, per)
        sets.append(s)
    return sets


class SpawriousDS(Dataset):
    def __init__(self, samples: Sequence[Sample], train: bool = False):
        import torchvision.transforms as T

        self.samples = samples
        aug = [T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip()] if train else [T.Resize(224), T.CenterCrop(224)]
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return self.tf(Image.open(s.path).convert("RGB")), s.label


def train_resnet18(train: List[Sample], device: str, epochs: int = 1, lr: float = 1e-4, wd: float = 1e-4,
                   batch_size: int = 64, num_workers: int = 8, out: str | Path | None = None) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, len(CLASSES)))
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    dl = DataLoader(SpawriousDS(train, train=True), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for ep in range(epochs):
        m.train()
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 50 == 0:
                log.info("epoch %d step %d/%d loss %.4f", ep, i, len(dl), loss.item())
    m.eval()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), out)
    return m


@torch.no_grad()
def extract_features(model: nn.Module, samples: List[Sample], device: str, batch_size: int = 128, num_workers: int = 8):
    feats, labels = [], []
    store = {}
    h = model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", o.flatten(1)))
    dl = DataLoader(SpawriousDS(samples), batch_size=batch_size, num_workers=num_workers)
    logits = []
    for x, y in dl:
        logits.append(model(x.to(device)).cpu())
        feats.append(store["f"].cpu())
        labels.append(y)
    h.remove()
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(logits).numpy()


# ---------------------------------------------------------------------------
# Logistic-regression experiments (App. F2)
# ---------------------------------------------------------------------------
def _logreg(kind: str, C: float = 1.0):
    """LogisticRegression across sklearn versions (penalty= was deprecated in 1.8 in favour of l1_ratio)."""
    import sklearn
    from sklearn.linear_model import LogisticRegression

    new = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
    if kind == "none":
        return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000) if new else \
            LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    return LogisticRegression(l1_ratio=1.0, C=C, solver="saga", max_iter=1000) if new else \
        LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=1000)


def _std(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others]


def fit_unregularized(Xtr, ytr, Xte, yte, units: Sequence[int]):
    u = list(units)
    a, b = _std(Xtr[:, u], Xte[:, u])
    clf = _logreg("none").fit(a, ytr)
    return float(clf.score(b, yte))


def l1_select(Xtr, ytr, n_units: int, Cs=None):
    """Find an l1 strength giving ~n_units non-zero units (any class). Returns (units, model, C)."""
    (a,) = _std(Xtr)
    lo, hi = 1e-4, 10.0
    best = None
    for _ in range(30):  # bisection on log C
        C = float(np.sqrt(lo * hi))
        clf = _logreg("l1", C).fit(a, ytr)
        nz = np.where(np.abs(clf.coef_).sum(0) > 0)[0]
        best = (nz, clf, C)
        if len(nz) == n_units:
            break
        if len(nz) > n_units:
            hi = C
        else:
            lo = C
    return best


def l1_eval(Xtr, ytr, Xte, yte, C):
    a, b = _std(Xtr, Xte)
    clf = _logreg("l1", C).fit(a, ytr)
    return float(clf.score(b, yte)), int((np.abs(clf.coef_).sum(0) > 0).sum())


def l1_tune_balanced(Xb, yb, Xte, yte, Cs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), n_splits=5, seed=0):
    """Hyper-parameter search with five 50-50 splits of the balanced set, then refit on all of it."""
    rng = np.random.default_rng(seed)
    scores = {}
    for C in Cs:
        accs = []
        for _ in range(n_splits):
            perm = rng.permutation(len(yb))
            h = len(perm) // 2
            acc, _ = l1_eval(Xb[perm[:h]], yb[perm[:h]], Xb[perm[h:]], yb[perm[h:]], C)
            accs.append(acc)
        scores[C] = float(np.mean(accs))
    bestC = max(scores, key=scores.get)
    acc, nz = l1_eval(Xb, yb, Xte, yte, bestC)
    return acc, nz, bestC
