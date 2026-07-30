"""Build a reduced star layer for phones.

The desktop map holds 10.8M objects and settles around 475 MB of JS heap. That is
comfortable on a laptop and it is not comfortable on a phone — iOS Safari starts
killing tabs somewhere around 300-400 MB, and the star layer alone is 7.4M of those
points and 57 MB of the download.

So phones get a thinned star layer rather than a thinned map. Every other catalogue
loads in full, because they are the part that makes this a map of the universe rather
than a map of the solar neighbourhood.

The subsample is uniform-random over the whole file, not a prefix. gaia_points.csv is
already in random_index order — Gaia's own shuffle — so a prefix would be defensible,
but relying on the input's ordering is the kind of assumption that silently breaks when
someone re-sorts the CSV. Drawing with a seeded RNG is explicit and costs nothing.

    ./.venv/bin/python make_mobile_stars.py     # writes gaia_points_lite.csv
    ./.venv/bin/python to_bin.py                # -> stars_lite.bin
    ./.venv/bin/python pack_octree.py           # -> stars_lite_o.bin
"""

import sys

import numpy as np
import pandas as pd

SRC = "gaia_points.csv"
OUT = "gaia_points_lite.csv"
TARGET = 1_200_000
SEED = 42


def main():
    df = pd.read_csv(SRC)
    n = len(df)
    if n <= TARGET:
        print(f"{SRC} already has {n:,} rows, at or under the {TARGET:,} target")
        df.to_csv(OUT, index=False)
        return 0

    rng = np.random.default_rng(SEED)
    keep = rng.choice(n, size=TARGET, replace=False)
    keep.sort()                      # keep file order stable for a readable diff
    lite = df.iloc[keep]

    # A uniform draw must not move the distribution. If it has, the sample is not
    # uniform and the phone would be shown a different galaxy to the desktop.
    r_full, r_lite = df["r"].to_numpy(), lite["r"].to_numpy()
    checks = [(p, float(np.percentile(r_full, p)), float(np.percentile(r_lite, p)))
              for p in (10, 50, 90, 99)]
    worst = max(abs(b - a) / max(a, 1e-9) for _, a, b in checks)

    print(f"{n:,} -> {len(lite):,} stars ({100 * len(lite) / n:.1f}%)")
    print("  radial distribution, full vs lite:")
    for p, a, b in checks:
        print(f"    p{p:<3} {a:8.1f} pc   {b:8.1f} pc   {100 * (b - a) / a:+.2f}%")
    print(f"  worst percentile drift {100 * worst:.2f}% "
          f"({'uniform' if worst < 0.01 else 'NOT UNIFORM — investigate'})")

    lite.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
