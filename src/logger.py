
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import src.config as cfg

cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = cfg.LOG_DIR / "app.log"

_FMT_FILE    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FMT_CONSOLE = "%(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fh = RotatingFileHandler(_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT_FILE))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_FMT_CONSOLE))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
