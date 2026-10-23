"""One heavy job at a time per container (stdlib only).

The supervisor holds this lock while any job runs, and selftest.py takes it too, so a self-test started
from the Render Shell waits for the worker's job instead of doubling the memory use.
"""
import contextlib
import fcntl
import os
import tempfile

LOCK_PATH = os.path.join(tempfile.gettempdir(), 'kronos_sniper_job.lock')


@contextlib.contextmanager
def job_lock():
    with open(LOCK_PATH, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
