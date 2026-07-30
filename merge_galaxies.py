"""Merge the two galaxy layers into one.

Why merge
---------
The map carried galaxies twice: a 43,507-point nearby layer and the 2.6M SDSS
spectroscopic deep field. Same class of object, drawn as two layers with two shell
radii (320 and 620), two sizes and two exposures — which is why one looked like a
different kind of thing to the other.

Merging is not only tidier, it is more accurate. Each layer was normalised into its own
shell, so the nearby galaxies were pushed out to 320 regardless of where they actually
sit. In one layer with one normalisation their true relative distance survives: the
nearby set reaches ~0.74e9 against the deep field's ~3.3e9, so it lands at roughly 22%
of the radius, which is where it belongs.

The nearby layer has no measured scalar — galaxies.bin is a headerless v1 file with no
provenance and no redshift column — so those points carry NaN and the shader draws them
neutral rather than inventing a colour. That is the existing contract for unmeasured
values, not a new exception.

Star thinning is NOT done here — see fetch_stars_bright.py, which re-pulls the layer
under a G < 13.26 magnitude limit. That yields 6,786,070 stars: a number the catalogue
produced rather than one picked, which is the point.

    ./.venv/bin/python merge_galaxies.py
    ./.venv/bin/python to_bin.py
    ./.venv/bin/python pack_octree.py
"""

import struct
import sys

import numpy as np
import pandas as pd

NEARBY_BIN = "galaxies.bin"          # v1: bare float32 [x,y,z,r], no header, no value
DEEP_CSV   = "sdss_galaxies_points.csv"
OUT_CSV    = "galaxies_all_points.csv"



def read_v1(path):
    raw = open(path, "rb").read()
    if raw[:4] == b"CSMS":
        _, n, stride = struct.unpack("<3I", raw[4:16])
        f = np.frombuffer(raw, dtype="<f4", offset=16, count=n * stride).reshape(n, stride)
        val = f[:, 4] if stride >= 5 else np.full(len(f), np.nan)
    else:
        f = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)
        val = np.full(len(f), np.nan)
    return pd.DataFrame({"x": f[:, 0].astype(float), "y": f[:, 1].astype(float),
                         "z": f[:, 2].astype(float), "r": f[:, 3].astype(float),
                         "redshift": val.astype(float)})


def main():
    near = read_v1(NEARBY_BIN)
    deep = pd.read_csv(DEEP_CSV)
    print(f"nearby {len(near):,} points, reach {near['r'].max():.4g}")
    print(f"deep   {len(deep):,} points, reach {deep['r'].max():.4g}")
    print(f"  -> nearby sits at {100 * near['r'].max() / deep['r'].max():.0f}% of the deep "
          f"field's radius, which one shared normalisation will now preserve")

    both = pd.concat([deep[["x", "y", "z", "r", "redshift"]],
                      near[["x", "y", "z", "r", "redshift"]]], ignore_index=True)
    measured = int(both["redshift"].notna().sum())
    both.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}: {len(both):,} galaxies "
          f"({measured:,} with a measured redshift, "
          f"{len(both) - measured:,} drawn neutral)")


    total = len(both) + 6_786_070 + 752_910 + 6_411
    print(f"\nmeasured objects once the bright star pull lands: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
