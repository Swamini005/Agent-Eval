"""
Logging configuration for the command-line entry points.

Library modules log through `logging.getLogger(__name__)` and configure nothing,
which is the rule a library has to follow: the application decides where output
goes. Before this, the pipeline wrote roughly twenty lines per task straight to
stdout -- fault injections, trace spans, plan steps -- so a 30-task run buried
its own summary under several hundred lines and there was no way to turn it off.

Default is WARNING: quiet unless something is wrong. `-v` adds the per-run
progress lines, `-vv` adds the per-task detail that used to be unconditional.
"""

import logging
import sys

FORMAT = "%(levelname)-8s %(name)s: %(message)s"


def configure(verbosity: int = 0) -> None:
    """
    Args:
        verbosity: 0 = warnings only, 1 = run progress (INFO),
            2 or more = per-task detail (DEBUG).
    """
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(FORMAT))

    root = logging.getLogger()
    # Replace rather than add, so repeated calls in a test session do not
    # duplicate every line.
    root.handlers[:] = [handler]
    root.setLevel(level)

    # These are third-party and chatty at DEBUG; raising their floor keeps -vv
    # showing this project's own output rather than HTTP wire logs.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "langsmith", "langfuse"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def verbosity_from_args(argv) -> int:
    """Count -v / --verbose flags, so callers need no argparse dependency."""
    count = 0
    for arg in argv:
        if arg in ("-v", "--verbose"):
            count += 1
        elif arg.startswith("-vv"):
            count += arg.count("v")
    return count
