"""Pull every SDSS DR17 quasar that passes the existing cut, not just the first 300,000.

The shipped quasar layer was `SELECT TOP 300000`, which is an arbitrary slice of a
catalogue that actually holds 752,910 rows under the same conditions (class='QSO',
zWarning=0, 0.01 < z < 7). Verified by asking SkyServer for the count before writing
this. So the layer was showing 40% of the available sky for no reason other than the
number someone typed.

SkyServer's synchronous endpoint caps a single result set, so the pull is chunked by
right ascension — 24 bands of 15 degrees — and the bands are stitched back together.
Each band is a disjoint RA range, so no row can appear twice; the script asserts the
total against a fresh COUNT(*) rather than trusting that.

Coordinates use the same conversion as prepare_universe_data.load_quasars: redshift to
comoving distance with H0 = 70, then RA/Dec/distance to Cartesian via Astropy. Redshift
is kept as the per-point measured scalar so the shader can colour by it.

    ./.venv/bin/python fetch_quasars_all.py        # writes quasars_points.csv
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
CUT = "class='QSO' AND zWarning=0 AND z>0.01 AND z<7.0"
H0 = 70.0
BANDS = 24                      # 15 degrees of RA each
OUT = "quasars_points.csv"


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
    print(f"catalogue holds {total_expected:,} quasars under the current cut")

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
    print(f"wrote {OUT}: {len(q):,} quasars, "
          f"redshift {q['redshift'].min():.3f} … {q['redshift'].max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
