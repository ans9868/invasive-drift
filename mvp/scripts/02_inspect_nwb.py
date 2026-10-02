#!/usr/bin/env python3
"""MVP step 2 — inspect ONE NWB session's structure (validate I/O before decoding).

Usage:  python 02_inspect_nwb.py <path-to.nwb>
"""
import sys

import h5py
from pynwb import NWBHDF5IO


def main(path: str) -> None:
    print(f"file: {path}\n")
    with h5py.File(path, "r") as fh:
        print("== top-level keys ==")
        for k in fh.keys():
            print("   ", k)
        for grp in ("acquisition", "processing", "intervals"):
            if grp in fh:
                print(f"== {grp} ==", list(fh[grp].keys()))

    io = NWBHDF5IO(path, "r")
    nwb = io.read()
    subj = nwb.subject.subject_id if nwb.subject is not None else None
    print(f"\nsession_id={nwb.session_id!r}  subject={subj!r}")

    if nwb.units is not None:
        print(f"units: {len(nwb.units)}   cols: {list(nwb.units.colnames)}")
    else:
        print("units: none")

    if nwb.trials is not None:
        df = nwb.trials.to_dataframe()
        print(f"trials: {df.shape}  cols: {list(df.columns)[:25]}")

    io.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python 02_inspect_nwb.py <path-to.nwb>")
    main(sys.argv[1])
