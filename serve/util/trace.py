"""Helix measurement-only request-path trace (2026-11-21). OFF unless BASALT_TRACE=<file>; never set it on a served
instance. Then every engine command written (">"), every engine line read ("<") and every authorized HTTP POST ("H") is
appended as `<perf_counter seconds> <thread name> <kind> <text>` (text: control characters replaced, first 100 chars).
Writes are block-buffered and flushed at exit, so the trace adds little to what it measures; it is still not free (one
lock + format per engine line): quote traced absolute times as traced, and compare traced runs only with traced runs."""
import atexit
import os
import re
import threading
import time

_path = os.environ.get("BASALT_TRACE")
_lock = threading.Lock()
_f = open(_path, "a", encoding="utf-8") if _path else None
_ctl = re.compile(r"[\x00-\x1f\x7f]")
if _f is not None:
    atexit.register(_f.flush)


def trace(kind: str, text: str) -> None:
    if _f is None:
        return
    line = _ctl.sub(" ", text[:100])
    with _lock:
        _f.write(f"{time.perf_counter():.6f} {threading.current_thread().name} {kind} {line}\n")
        if kind != "<":   # commands and requests are rare: flush, so a force-killed server keeps all but its last lines
            _f.flush()
