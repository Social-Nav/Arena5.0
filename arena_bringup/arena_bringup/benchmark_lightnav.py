"""Compatibility entry point for the shared external-client deployment driver."""
import sys
import signal
import subprocess
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_bringup.benchmark_external import Deployment, installed_features, main
if __name__ == '__main__':
    def interrupt(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'[lightnav] ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
