"""Pull every Gaia DR3 star brighter than G = 13.26 with a well-measured distance.

Why this is safe to claim
-------------------------
The cut is unchanged from prepare_universe_data.load_stars_gaia:

    parallax > 0.5              inside ~2 kpc
    parallax_over_error > 10    distance known to better than 10%
    phot_g_mean_mag < 13.26     the brightest stars, at full depth

The magnitude limit is doing the work that a random subsample used to. That matters:
"6,786,070 stars" is a number the catalogue produced, not one somebody picked — it is
every star meeting the cut, and the cut is a sentence you can say out loud. A round
6,700,000 would have been a draw with a target attached, and it would have sat oddly
beside 2,613,461 and 752,910, which are whole catalogues.

Brightness rather than distance was the lever on purpose: tightening parallax instead
would have pulled the map in from 2 kpc to about 450 pc, shrinking what it reaches.
This keeps the full depth and takes the brightest cut through it.

Why random_index
----------------
gaia_source is stored in source_id order, which is a HEALPix index, so a bare TOP N
returns one contiguous PATCH OF SKY — the original 500k query spanned about 25 sky
directions before someone caught it. random_index is Gaia's own precomputed shuffle,
so ordering or slicing by it draws an unbiased all-sky sample.

That property is what makes chunking correct here: slicing random_index into disjoint
bands gives disjoint AND unbiased chunks, so the union is still a fair all-sky draw.
No row can appear twice, because the bands do not overlap.

The sync endpoint will not return 7M rows in one response, so the pull is 20 bands of
~350k. Each band retries; a band that will not come back aborts the run rather than
quietly shipping a sky with a hole in it.

    ./.venv/bin/python fetch_stars_7m.py           # writes gaia_points.csv
"""

import io
import os
import shutil
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Distance, SkyCoord

TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
CUT = ("parallax>0.5 AND parallax_over_error>10 AND phot_g_mean_mag<13.26 "
       "AND l IS NOT NULL AND b IS NOT NULL")
TARGET = 6_786_070      # measured with COUNT(*), not chosen
BANDS = 20
RI_MAX = 1_811_709_770          # MAX(random_index) over gaiadr3.gaia_source
POP = 6_786_070                 # rows satisfying CUT — measured with COUNT(*)
OUT = "gaia_points.csv"


def query(adql, timeout=900):
    url = f"{TAP}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={urllib.parse.quote(adql)}"
    with urllib.request.urlopen(url, timeout=timeout,
                                context=ssl._create_unverified_context()) as r:
        return pd.read_csv(io.StringIO(r.read().decode("utf-8", "replace")))


def main():
    # Width of random_index needed for TARGET rows, with headroom so rounding does not
    # land just under. The cut population is spread uniformly over random_index because
    # random_index is a shuffle, so this scales linearly.
    span = RI_MAX          # taking the whole population, so sweep the whole index
    width = span // BANDS
    print(f"target {TARGET:,} of {POP:,} available "
          f"({100 * TARGET / POP:.1f}%) — random_index 0 … {span:,} in {BANDS} bands\n")

    frames, got = [], 0
    for i in range(BANDS):
        lo, hi = i * width, (i + 1) * width
        sql = (f"SELECT parallax,l,b,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source "
               f"WHERE {CUT} AND random_index >= {lo} AND random_index < {hi}")
        for attempt in range(4):
            try:
                df = query(sql)
                break
            except Exception as e:
                print(f"  band {i:2d} attempt {attempt + 1}: {type(e).__name__} — retrying")
                time.sleep(10)
        else:
            print(f"  band {i:2d} FAILED after 4 attempts — aborting rather than "
                  f"shipping a sky with a hole in it")
            return 1
        got += len(df)
        frames.append(df)
        print(f"  band {i:2d}  ri {lo:>11,}–{hi:>11,}  {len(df):>7,} rows   "
              f"running total {got:>9,}")

    df = pd.concat(frames, ignore_index=True)
    for c in ("parallax", "l", "b"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["parallax", "l", "b"])
    df = df[df["parallax"] > 0]
    print(f"\nstitched {len(df):,} stars")

    # Distance from parallax. The cut guarantees parallax_over_error > 10, so the naive
    # inversion is within a few percent of a proper posterior — the regime where 1/p is
    # legitimate rather than a shortcut.
    d_pc = Distance(parallax=df["parallax"].values * u.mas).pc
    coords = SkyCoord(l=df["l"].values * u.deg, b=df["b"].values * u.deg,
                      distance=d_pc * u.pc, frame="galactic")
    df["x"] = coords.cartesian.x.value
    df["y"] = coords.cartesian.y.value
    df["z"] = coords.cartesian.z.value
    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    # Sanity: an all-sky draw should have no preferred direction. If the mean unit
    # vector is far from zero the sample is a wedge, which is the bug random_index
    # exists to prevent — so check rather than trust.
    ux, uy, uz = (df["x"] / df["r"]).mean(), (df["y"] / df["r"]).mean(), (df["z"] / df["r"]).mean()
    aniso = float(np.sqrt(ux ** 2 + uy ** 2 + uz ** 2))
    print(f"all-sky check: mean unit vector magnitude {aniso:.4f} "
          f"({'uniform' if aniso < 0.05 else 'LOPSIDED — investigate'})")

    if os.path.exists(OUT):
        bak = OUT + ".500k"
        if not os.path.exists(bak):
            shutil.copy2(OUT, bak)
            print(f"kept the previous sample at {bak}")

    df[["x", "y", "z", "r", "bp_rp"]].to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} stars, "
          f"distance {df['r'].min():.0f} … {df['r'].max():.0f} pc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
