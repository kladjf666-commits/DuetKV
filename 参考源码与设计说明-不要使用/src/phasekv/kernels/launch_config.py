"""Compatibility entry; edit PhaseKV/kernels/launch_config.py instead."""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import runpy
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("PhaseKV.kernels.launch_config", run_name="__main__")
else:
    import sys
    from importlib import import_module
    sys.modules[__name__] = import_module("PhaseKV.kernels.launch_config")
