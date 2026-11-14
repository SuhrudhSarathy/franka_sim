"""Make Ctrl+C work in long-running loops, including under ``mjpython``.

``mjpython`` (needed for MuJoCo's viewer on macOS) starts the script with SIGINT ignored, so
Ctrl+C in the terminal would do nothing. ``interruptible()`` restores Python's default handler
(raise ``KeyboardInterrupt``) for the duration of a block, when called from the main thread.
"""

from __future__ import annotations

import contextlib
import signal
import threading


@contextlib.contextmanager
def interruptible():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
