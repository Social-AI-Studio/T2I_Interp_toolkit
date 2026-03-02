"""Package-level helpers for locating installed config files.

Usage inside a script::

    from config._hydra_config import config_dir
    @hydra.main(config_path=config_dir("steer"), config_name="run", version_base=None)
    def main(cfg): ...
"""
import importlib.resources as _ir
import pathlib


def config_dir() -> str:
    """Return the absolute path to the installed config root dir.

    Works whether the package is installed as a wheel (importlib.resources)
    or used directly from the source tree (editable install / ``pip install -e .``).
    """
    # importlib.resources path — works for both installed and editable
    pkg_ref = _ir.files("t2i_interp.config")
    return str(pathlib.Path(str(pkg_ref)).resolve())
