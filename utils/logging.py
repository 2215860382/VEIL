"""Tiny logging helper."""
import logging
import os
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str = "veil", level: str | None = None) -> logging.Logger:
    level = level or os.environ.get("VEIL_LOG", "INFO")
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger
