from __future__ import annotations

import logging
import re

from agentlings.log import setup_logging


def test_handler_writes_to_stderr() -> None:
    import sys

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    setup_logging("DEBUG")
    handler = root.handlers[-1]
    assert handler.stream is sys.stderr  # type: ignore[attr-defined]


def test_log_format_pattern() -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    setup_logging("DEBUG")
    handler = root.handlers[-1]
    formatter = handler.formatter
    assert formatter is not None

    record = logging.LogRecord(
        name="agentlings.store",
        level=logging.INFO,
        pathname="store.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - agentlings\.store - test message"
    assert re.match(pattern, formatted), f"Format mismatch: {formatted}"


def test_log_level_respected() -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    setup_logging("WARNING")
    assert root.level == logging.WARNING
