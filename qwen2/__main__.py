"""Compatibility entry; edit PhaseKV/qwen2/__main__.py instead."""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import runpy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("PhaseKV.qwen2.__main__", run_name="__main__")
else:
    import sys
    from importlib import import_module
    sys.modules[__name__] = import_module("PhaseKV.qwen2.__main__")
