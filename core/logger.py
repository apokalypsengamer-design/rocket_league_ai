import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        log.addHandler(ch)

        filename = os.path.join(log_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{name}.log")
        fh = logging.FileHandler(filename)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log
