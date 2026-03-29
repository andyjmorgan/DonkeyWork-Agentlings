"""Logging configuration for agentlings."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a stderr stream handler.

    Args:
        level: Log level name (e.g. "INFO", "DEBUG").
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
