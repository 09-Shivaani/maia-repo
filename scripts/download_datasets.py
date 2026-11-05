#!/usr/bin/env python
"""Download every dataset used by the reproduction and the extensions (real data, no stubs).

  --cc3m N            CC3M subset (Sharma et al. 2018) via img2dataset - real-image pool for dataset
                      exemplars (replaces ImageNet val) and synthetic-neuron exemplars. Many CC3M URLs are
                      dead, so ask for ~30% more than you need.
  --spawrious         Spawrious O2O-Easy (Lynch et al. 2023), official 224px tarball
  --bold              BOLD prompts (Dhamala et al. 2021) from github.com/amazon-science/bold
  --stereoset         StereoSet dev set (Nadeem et al. 2021) from github.com/moinnadeem/StereoSet
  --text-corpus       WikiText-103 train paragraphs (real corpus for LLM dataset exemplars)
  --concepts          CLIP-Dissect 20k concept list (baseline)
  --nltk              WordNet data for synthetic text neurons
  --bills-explanations GPT-2 XL neuron explanations of Bills et al. (2023), reference labels for MAIA-NLP
  --mib               pre-cache MIB IOI (Mueller et al. 2025) from the HF hub
  --all               everything above (CC3M size from --cc3m, default 130000)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from _common import ROOT, parser, setup

SPAWRIOUS_URL = "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious224__o2o_easy.tar.gz?dl=1"
BOLD_GIT = "https://github.com/amazon-science/bold.git"
STEREOSET_RAW = "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json"
CONCEPTS_RAW = "https://raw.githubusercontent.com/Trustworthy-ML-Lab/CLIP-dissect/main/data/20k.txt"
BILLS_BASE = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/explanations"


def _fetch(url: str, dst: Path, desc: str = ""):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {desc or url} -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "maia-repo-downloader"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def cc3m(root: Path, n: int, processes: int, threads: int, seed: int = 0):
    from datasets import load_dataset

    out = root / "cc3m"
    meta = out / "cc3m_subset.parquet"
    if not meta.exists():
        print(f"loading CC3M caption/URL list and sampling {n} rows ...")
        ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(meta))
    cmd = [sys.executable, "-m", "img2dataset", "--url_list", str(meta), "--input_format", "parquet",
           "--url_col", "image_url", "--caption_col", "caption", "--output_format", "files",
           "--output_folder", str(out / "images"), "--image_size", "256", "--resize_mode", "keep_ratio",
           "--processes_count", str(processes), "--thread_count", str(threads), "--retries", "1",
           "--timeout", "10", "--enable_wandb", "False"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    n_ok = sum(1 for _ in (out / "images").rglob("*.jpg"))
    print(f"CC3M: {n_ok} images downloaded into {out / 'images'}")


def spawrious(root: Path):
    out = root / "spawrious"
    if (out / "spawrious224").exists():
        print("Spawrious already present")
        return
    tar = out / "spawrious224__o2o_easy.tar.gz"
    if not tar.exists():
        _fetch(SPAWRIOUS_URL, tar, "Spawrious O2O-Easy (~several GB)")
    print("extracting ...")
    with tarfile.open(tar) as t:
        t.extractall(out)
    tar.unlink()


def bold(root: Path):
    out = root / "bold"
    if (out / "prompts").exists():
        print("BOLD already present")
        return
    subprocess.run(["git", "clone", "--depth", "1", BOLD_GIT, str(out)], check=True)


def stereoset(root: Path):
    _fetch(STEREOSET_RAW, root / "stereoset" / "dev.json", "StereoSet dev")


def concepts(root: Path):
    _fetch(CONCEPTS_RAW, root / "concepts" / "20k.txt", "CLIP-Dissect 20k concepts")


def text_corpus(root: Path, n_par: int):
    from datasets import load_dataset

    out = root / "text" / "wikitext103_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    k = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            t = row["text"].strip()
            if len(t) < 80 or t.startswith("="):
                continue
            f.write(json.dumps({"text": t[:600]}) + "\n")
            k += 1
            if k >= n_par:
                break
    print(f"wrote {k} paragraphs to {out}")


def nltk_data():
    import nltk

    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg)


def bills(root: Path, layers=range(0, 48, 4), neurons=range(0, 6400, 64)):
    """A grid of GPT-2 XL neuron explanations (layer, neuron) from Bills et al. 2023."""
    out = root / "bills_explanations"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for l in layers:
        for n in neurons:
            url = f"{BILLS_BASE}/{l}/{n}.jsonl"
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    d = json.loads(r.read().decode().splitlines()[0])
                sc = d["scored_explanations"][0]
                rows.append({"layer": l, "neuron": n, "explanation": sc["explanation"], "score": sc.get("scored_simulation", {}).get("ev_correlation_score")})
            except Exception as e:  # some neurons are missing
                print(f"skip {l}/{n}: {e}")
    (out / "gpt2-xl.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"saved {len(rows)} explanations")


def mib():
    from datasets import load_dataset

    for split in ("train", "validation", "test"):
        try:
            load_dataset("mib-bench/ioi", split=split)
        except Exception as e:
            print(f"MIB ioi {split}: {e}")


def main():
    p = parser(__doc__)
    p.add_argument("--cc3m", type=int, default=0)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--threads", type=int, default=64)
    for flag in ("spawrious", "bold", "stereoset", "text_corpus", "concepts", "nltk", "bills_explanations", "mib", "all"):
        p.add_argument("--" + flag.replace("_", "-"), action="store_true")
    p.add_argument("--corpus-size", type=int, default=200000)
    args = p.parse_args()
    cfg = setup(args)
    root = ROOT / cfg["paths"]["data_root"]
    if args.all:
        args.cc3m = args.cc3m or 130000
        for f in ("spawrious", "bold", "stereoset", "text_corpus", "concepts", "nltk", "bills_explanations", "mib"):
            setattr(args, f, True)
    if args.bold:
        bold(root)
    if args.stereoset:
        stereoset(root)
    if args.concepts:
        concepts(root)
    if args.nltk:
        nltk_data()
    if args.text_corpus:
        text_corpus(root, args.corpus_size)
    if args.bills_explanations:
        bills(root)
    if args.mib:
        mib()
    if args.spawrious:
        spawrious(root)
    if args.cc3m:
        cc3m(root, args.cc3m, args.processes, args.threads)


if __name__ == "__main__":
    main()
