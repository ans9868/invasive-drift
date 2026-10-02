#!/usr/bin/env python3
"""MVP step 0 — verify the environment (python + required imports)."""
import importlib
import sys

REQ = ["numpy", "scipy", "sklearn", "matplotlib", "pandas", "pynwb", "h5py", "dandi"]


def main() -> int:
    print("python:", sys.version.split()[0])
    missing = []
    for m in REQ:
        try:
            mod = importlib.import_module(m)
            print(f"  ok       {m:12s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  MISSING  {m:12s} ({exc})")
            missing.append(m)
    if missing:
        print("\nInstall missing:  pip install " + " ".join(missing))
        return 1
    print("\nENV OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
