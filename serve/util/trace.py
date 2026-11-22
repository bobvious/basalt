"""Helix measurement-only request-path trace (2026-11-21). Off unless BASALT_TRACE=<file>: then every engine command
written (">"), every engine line read ("<") and every HTTP POST arriving ("H") is appended as
`<perf_counter seconds> <thread> <kind> <first 100 chars>`. Exists to split a request's time-to-first-token into stages."""
import os
import threading
import time

_path = os.environ.get("BASALT_TRACE")
_lock = threading.Lock()
_f = open(_path, "a", encoding="utf-8", buffering=1) if _path else None


def trace(kind: str, text: str) -> None:
    if _f is None:
        return
    line = text.replace("\n", "\\n")[:100]
    with _lock:
        _f.write(f"{time.perf_counter():.6f} {threading.current_thread().name} {kind} {line}\n")
