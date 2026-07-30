"""Recompute redshift-derived positions with real cosmology instead of a low-z expansion.

The problem
-----------
Every redshift layer placed points with a second-order expansion:

    d = (c/H0) * z * (1 + z*(-0.5 + 0.167*z))

That is a Taylor series around z = 0. It is fine to z ~ 0.5 and degrades fast after,
because it has no idea the universe stopped decelerating. Measured against
FlatLambdaCDM(H0=70, Om0=0.3):

    z = 0.5      -10.2%
    z = 1.0      -13.5%
    z = 2.0      +10.5%
    z = 3.0     +102.9%
    z = 5.0     +637.2%
    z = 7.0    +1876.1%

The galaxy layer is cut at z < 1, so it is wrong by at most ~14% — visible but not
absurd. The quasar layer runs to z = 7, and its median is 1.66 with p99 at 3.76, so
HALF that layer is misplaced by more than 10% and the tail is thrown out by up to 19x.
That is what produced a quasar layer whose maximum radius was seven times its own p99:
the far points were not distant, they were wrong.

Redshift is stored per point in the CSVs, so the fix needs no refetch — only the
radial coordinate changes. Direction is untouched: it comes from RA/Dec, which was
never in question.

Comoving distance is the right quantity here. It is what "where is it now" means on a
map of large-scale structure, and it is what makes the filament layer and the galaxy
layer commensurable.

    ./.venv/bin/python fix_comoving_distance.py
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
H0 = 70.0
FILES = ["quasars_points.csv", "sdss_galaxies_points.csv"]


def old_distance(z):
    """The expansion that was used, kept so the change can be reported honestly."""
    return (3e5 / H0) * z * (1 + z * (-0.5 + z * 0.167))


def main():
    for path in FILES:
        if not os.path.exists(path):
            print(f"skip {path} (absent)")
            continue

        df = pd.read_csv(path)
        if "redshift" not in df.columns:
            print(f"skip {path}: no redshift column, cannot recompute")
            continue

        z = df["redshift"].to_numpy(dtype=float)
        r_old = df["r"].to_numpy(dtype=float)

        # Direction is preserved exactly; only the radius is replaced. Guard against a
        # zero-radius row so the unit vector stays finite.
        safe = np.where(r_old > 0, r_old, 1.0)
        ux, uy, uz = (df["x"].to_numpy() / safe,
                      df["y"].to_numpy() / safe,
                      df["z"].to_numpy() / safe)

        # Interpolate the comoving distance over a z grid rather than calling the
        # integral 2.6M times — the grid is dense enough that interpolation error is
        # far below the quantisation the octree applies later.
        grid = np.linspace(max(z.min(), 1e-6), z.max(), 4000)
        d_grid = COSMO.comoving_distance(grid).value          # Mpc
        d_new = np.interp(z, grid, d_grid) * 1e6              # viewer units

        d_old_mpc = old_distance(z)
        shifted = np.abs(d_new / 1e6 - d_old_mpc) / np.maximum(d_old_mpc, 1e-9)

        df["x"], df["y"], df["z"] = ux * d_new, uy * d_new, uz * d_new
        df["r"] = d_new

        backup = path + ".preCosmology"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)

        df.to_csv(path, index=False)
        print(f"{path}: {len(df):,} points")
        print(f"    z          {z.min():.4f} … {z.max():.3f}  (median {np.median(z):.3f})")
        print(f"    max radius {r_old.max():.4g} -> {d_new.max():.4g}")
        print(f"    median point moved {np.median(shifted) * 100:.1f}%, "
              f"worst {shifted.max() * 100:.0f}%")

    print("\nRe-run to_bin.py then pack_octree.py to rebuild the octrees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
