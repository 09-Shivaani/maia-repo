"""Shared CLI plumbing for scripts/*.py."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maia.config import load_config  # noqa: E402


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="task YAML in configs/ (merged over default.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup(args) -> dict:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "openai", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return load_config(args.config, args.set)
