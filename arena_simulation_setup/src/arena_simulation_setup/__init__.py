import os
from pathlib import Path

ASS_DIR: Path
AB_DIR: Path


def _resolve_default_assets_dir() -> Path:
    """Return the Arena asset repository root.

    Historically this defaulted to ``Path.cwd() / '_assets'``.  That only works
    when processes are launched from the Arena source checkout.  In the dual
    Docker Isaac evaluation the task generator is launched from the workspace
    root (``/home/ubuntu/arena_jazzy_ws``), while the hospital assets live under
    ``src/Arena/_assets``.  Falling back to cwd therefore makes identifiers like
    ``Hospital/SM_ReceptionDesk_01a`` resolve under the non-existent
    ``/home/ubuntu/arena_jazzy_ws/_assets`` and the hospital degrades to only
    procedural floors/walls/doors.
    """
    explicit = os.environ.get('ARENA_ASSETS_DIR')
    if explicit:
        return Path(explicit)

    cwd_candidate = Path(os.getcwd()) / '_assets'
    if cwd_candidate.exists():
        return cwd_candidate

    package_path = Path(__file__).resolve()
    for parent in package_path.parents:
        candidates = (
            parent / '_assets',
            parent / 'src' / 'Arena' / '_assets',
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return cwd_candidate


ARENA_ASSETS_DIR: Path = _resolve_default_assets_dir()
DOMAIN_DEFAULT: str = 'Common'


try:
    import ament_index_python.packages
    ASS_DIR = ament_index_python.packages.get_package_share_path('arena_simulation_setup')
    AB_DIR = ament_index_python.packages.get_package_share_path('arena_bringup')
except ImportError:
    ASS_DIR = Path(os.environ.get('ASS_DIR', 'arena_simulation_setup'))
    AB_DIR = Path(os.environ.get('AB_DIR', 'arena_bringup'))
