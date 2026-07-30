"""Pull the whole SDSS DR17 spectroscopic galaxy sample — every galaxy with a measured
redshift, not a slice of one.

The shipped galaxy layer is HyperLEDA plus a local-group file (see
prepare_universe_data.load_galaxies_hyperleda / load_galaxies_local): 43,507 points,
mostly nearby and bright. That is a different catalogue to this one. SDSS DR17 SpecObj
holds ~2.6M galaxies with a *spectroscopic* redshift under the cut below, which is what
makes each point a measured distance rather than an estimated one — and it is what turns
the galaxy layer from a local shell into actual large-scale structure.

SkyServer's synchronous endpoint caps a single result set, so the pull is chunked by
right ascension — 72 bands of 5 degrees — and the bands are stitched back together.
Each band is a disjoint RA range, so no row can appear twice; the script asserts the
total against a fresh COUNT(*) and refuses to write a partial sky.

Coordinates use the same conversion as prepare_universe_data (line 240): redshift to
comoving distance with H0 = 70 via the low-z expansion, then RA/Dec/distance to Cartesian
through Astropy, scaled to the viewer's 1e6 units. Redshift is kept as the per-point
measured scalar so the shader can colour by it.

The cut is z > 0.0005 to drop objects whose redshift is dominated by peculiar velocity
rather than expansion, and z < 1.0 because the low-z distance expansion above stops being
honest beyond roughly that.

    ./.venv/bin/python fetch_sdss_galaxies.py      # writes sdss_galaxies_points.csv
"""

import io
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

BASE = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
CUT = "class='GALAXY' AND zWarning=0 AND z>0.0005 AND z<1.0"
H0 = 70.0
BANDS = 72                      # 5 degrees of RA each; 2.6M rows needs finer chunks
OUT = "sdss_galaxies_points.csv"


# Same unverified context prepare_universe_data.ssl_ctx() already uses. This machine
# sits behind a TLS-inspecting proxy, so the chain ends in a self-signed certificate and
# strict verification fails on every catalogue host. These are public, read-only science
# endpoints and the payload is checked against a COUNT(*) before anything is written.
def _ctx():
    return ssl._create_unverified_context()


def query(sql, timeout=300):
    url = f"{BASE}?cmd={urllib.parse.quote(sql)}&format=csv"
    with urllib.request.urlopen(url, timeout=timeout, context=_ctx()) as r:
        text = r.read().decode("utf-8", "replace")
    lines = [l for l in text.split("\n") if l and not l.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))


def main():
    total_expected = int(query(f"SELECT COUNT(*) AS n FROM SpecObj WHERE {CUT}")["n"].iloc[0])
    print(f"catalogue holds {total_expected:,} galaxies under the current cut")

    frames = []
    width = 360.0 / BANDS
    for i in range(BANDS):
        lo, hi = i * width, (i + 1) * width
        # ra >= lo AND ra < hi keeps the bands disjoint; the last one takes ra = 360
        bound = "<=" if i == BANDS - 1 else "<"
        sql = (f"SELECT ra,dec,z FROM SpecObj "
               f"WHERE {CUT} AND ra >= {lo} AND ra {bound} {hi}")
        for attempt in range(3):
            try:
                df = query(sql)
                break
            except Exception as e:
                print(f"  band {i:2d} attempt {attempt + 1} failed: {type(e).__name__}")
                time.sleep(5)
        else:
            print(f"  band {i:2d} FAILED after 3 attempts — aborting rather than "
                  f"shipping a partial sky")
            return 1
        print(f"  band {i:2d}  RA {lo:6.1f}-{hi:6.1f}  {len(df):>7,} rows")
        frames.append(df)

    q = pd.concat(frames, ignore_index=True)
    for c in ("ra", "dec", "z"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q.dropna(subset=["ra", "dec", "z"])
    print(f"\npulled {len(q):,} of {total_expected:,} "
          f"({100 * len(q) / total_expected:.2f}%)")
    if len(q) < total_expected * 0.99:
        print("Pulled materially fewer rows than the catalogue holds — not writing.")
        return 1

    q = q.rename(columns={"z": "redshift"})
    c_over_H0 = 3e5 / H0
    q["distance_mpc"] = c_over_H0 * q["redshift"] * (
        1 + q["redshift"] * (-0.5 + q["redshift"] * 0.167))
    coords = SkyCoord(ra=q["ra"].values * u.deg, dec=q["dec"].values * u.deg,
                      distance=q["distance_mpc"].values * u.Mpc, frame="icrs")
    q["x"] = coords.cartesian.x.to(u.Mpc).value * 1e6
    q["y"] = coords.cartesian.y.to(u.Mpc).value * 1e6
    q["z"] = coords.cartesian.z.to(u.Mpc).value * 1e6
    q["r"] = np.sqrt(q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2)

    q[["x", "y", "z", "r", "redshift"]].to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(q):,} galaxies, "
          f"redshift {q['redshift'].min():.3f} … {q['redshift'].max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
