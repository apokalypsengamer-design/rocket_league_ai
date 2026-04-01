from __future__ import annotations
import logging
import os
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
) -> logging.Logger:
    """
    Return a named logger that writes to both console and a dated log file.

    Console handler: INFO and above (human-readable status).
    File handler:    DEBUG and above (full trace for post-mortem analysis).

    Calling setup_logger() multiple times with the same name is safe —
    handlers are only added once.
    """
    os.makedirs(log_dir, exist_ok=True)
    log = logging.getLogger(name)

    if log.handlers:
        return log   # already configured

    numeric = getattr(logging, level.upper(), logging.INFO)
    log.setLevel(logging.DEBUG)   # capture everything; handlers filter

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(name)-12s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console — INFO+
    ch = logging.StreamHandler()
    ch.setLevel(numeric)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File — DEBUG+
    log_file = os.path.join(log_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{name}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def print_header() -> None:
    """Print a readable column header for the live status lines from Brain."""
    cols = (
        f"{'Frame':>7}  "
        f"{'Ball (x,y) Seite':<22}  "
        f"{'Boost':>7}  "
        f"{'Phase':<15}  "
        f"{'Action':<35}  "
        "Grund"
    )
    sep = "─" * min(len(cols) + 20, 180)
    print(f"\n{sep}\n{cols}\n{sep}")
