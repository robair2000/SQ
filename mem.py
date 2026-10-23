"""Memory readings for the logs: this process, the largest child so far, and the container total."""
import resource


def _read(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v == 'max' else int(v)
    except Exception:
        return None


def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


def report():
    parts = []
    r = rss_mb()
    if r is not None:
        parts.append(f"worker {r:.0f} MB")
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    if child:
        parts.append(f"largest child peak {child:.0f} MB")
    cur = _read('/sys/fs/cgroup/memory.current') or _read('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    peak = _read('/sys/fs/cgroup/memory.peak') or _read('/sys/fs/cgroup/memory/memory.max_usage_in_bytes')
    limit = _read('/sys/fs/cgroup/memory.max') or _read('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if cur:
        s = f"container {cur / 2**20:.0f}"
        if peak:
            s += f" (peak {peak / 2**20:.0f})"
        if limit and limit < 2**50:
            s += f" / {limit / 2**20:.0f}"
        parts.append(s + " MB")
    return 'memory: ' + ', '.join(parts) if parts else 'memory: n/a'
