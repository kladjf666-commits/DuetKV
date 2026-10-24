"""Compatibility entry; edit KIVI/legacy_fake_quant.py instead."""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import runpy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("KIVI.legacy_fake_quant", run_name="__main__")
else:
    import sys
    from importlib import import_module
    sys.modules[__name__] = import_module("KIVI.legacy_fake_quant")
