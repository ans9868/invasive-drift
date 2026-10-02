#!/usr/bin/env python3
"""MVP step 1 — list available sessions in the candidate datasets (no heavy download).

Uses the DANDI API only (metadata), so it is cheap.
"""
from dandi.dandiapi import DandiAPIClient

DANDISETS = {
    "FALCON M1-A (macaque M1 reach-to-grasp)": "000941",
    "FALCON M1-B (second monkey)": "001209",
    "Perich & Miller long-term reaching": "000688",
}
N_PREVIEW = 8


def main() -> None:
    with DandiAPIClient() as client:
        for name, ds in DANDISETS.items():
            print(f"== {name}  (DANDI:{ds}) ==")
            try:
                dandiset = client.get_dandiset(ds, "draft")
                paths = sorted(a.path for a in dandiset.get_assets())
            except Exception as exc:  # noqa: BLE001
                print("   error:", exc)
                continue
            print(f"   {len(paths)} assets")
            for p in paths[:N_PREVIEW]:
                print("     ", p)
            if len(paths) > N_PREVIEW:
                print("      ...")
            print()


if __name__ == "__main__":
    main()
