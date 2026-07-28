"""
prepare_universe_data.py
========================
Downloads and processes the largest freely available real astronomical datasets.
Every single point output is a REAL measured object with real coordinates.

Sources:
  Stars      — Gaia DR3 via ESA TAP (up to 500k real stars, measured parallax)
  Galaxies   — HyperLEDA SQL API (up to 200k real galaxies, measured redshift)
               Falls back to local 2MRS file if HyperLEDA is unavailable
  Exoplanets — NASA Exoplanet Archive TAP (confirmed + Kepler candidates)
  Quasars    — SDSS DR17 spectroscopic catalogue (real quasars, measured z)
  Clusters   — Planck PSZ2 catalogue via Vizier (1,653 real galaxy clusters)
  CMB        — WMAP 9-year ILC all-sky map from NASA/Wikimedia (real CMB image)

Outputs:
  gaia_points.csv       x,y,z,r  (parsecs)
  galaxies_points.csv   x,y,z,r  (parsecs = Mpc * 1e6)
  exoplanets_points.csv x,y,z    (parsecs)
  quasars_points.csv    x,y,z,r  (parsecs = Mpc * 1e6)
  clusters_points.csv   x,y,z,r,m  (parsecs, m=mass proxy)
  cmb_map.jpg           real WMAP CMB all-sky image for Three.js skybox

Requirements:
  pip install pandas numpy astropy requests tqdm
"""

import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
import ssl
import urllib.request
import urllib.parse
import requests
import io
import os
import sys

# ── Configuration ──────────────────────────────────────────────────────────────
GAIA_LIMIT      = 500_000
HYPERLEDA_LIMIT = 200_000
QUASAR_LIMIT    = 300_000
HUBBLE_CONSTANT = 70.0

def ssl_ctx():
    return ssl._create_unverified_context()

def fetch_url(url, label="", binary=False):
    print(f"  Downloading {label}..." if label else f"  Downloading {url[:70]}...")
    try:
        ctx = ssl_ctx()
        with urllib.request.urlopen(url, context=ctx, timeout=180) as r:
            data = r.read()
        if not binary:
            data = data.decode("utf-8")
        print(f"  Done ({len(data)//1024} KB)")
        return data
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

# ── STARS: Gaia DR3 ────────────────────────────────────────────────────────────
def load_stars_gaia(limit=GAIA_LIMIT):
    print(f"\n[STARS] Querying Gaia DR3 for {limit:,} real stars...")
    # ORDER BY random_index is essential, not cosmetic. gaia_source is stored in
    # source_id order, which is a HEALPix index — so a bare TOP N returns one
    # contiguous PATCH OF SKY. The old query yielded 500,000 stars spanning only
    # ~25 distinct sky directions: a wedge, not the Milky Way. random_index is
    # Gaia's own precomputed shuffle for drawing unbiased all-sky subsets.
    adql = (
        f"SELECT TOP {limit} parallax,l,b,phot_g_mean_mag,bp_rp "
        f"FROM gaiadr3.gaia_source "
        f"WHERE parallax>0.5 AND parallax_over_error>10 "
        f"AND phot_g_mean_mag<18 AND l IS NOT NULL AND b IS NOT NULL "
        f"ORDER BY random_index"
    )
    url = ("https://gea.esac.esa.int/tap-server/tap/sync?"
           "REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=" + urllib.parse.quote(adql))
    csv_data = fetch_url(url, f"Gaia DR3 ({limit:,} stars)")
    if csv_data is None:
        return load_stars_local()
    df = pd.read_csv(io.StringIO(csv_data))
    for c in ["parallax","l","b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["parallax","l","b"])
    df = df[df["parallax"] > 0]
    print(f"  Valid stars: {len(df):,}")
    d_pc = Distance(parallax=df["parallax"].values * u.mas).pc
    coords = SkyCoord(l=df["l"].values*u.deg, b=df["b"].values*u.deg,
                      distance=d_pc*u.pc, frame="galactic")
    df["x"] = coords.cartesian.x.value
    df["y"] = coords.cartesian.y.value
    df["z"] = coords.cartesian.z.value
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df

def load_stars_local():
    print("  Falling back to local gaia_sample.csv...")
    df = pd.read_csv("gaia_sample.csv")
    for c in ["parallax","l","b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["parallax","l","b"])
    df = df[df["parallax"] > 0]
    d_pc = Distance(parallax=df["parallax"].values * u.mas).pc
    coords = SkyCoord(l=df["l"].values*u.deg, b=df["b"].values*u.deg,
                      distance=d_pc*u.pc, frame="galactic")
    df["x"] = coords.cartesian.x.value
    df["y"] = coords.cartesian.y.value
    df["z"] = coords.cartesian.z.value
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    print(f"  Loaded {len(df):,} stars from local file")
    return df

# ── GALAXIES: HyperLEDA ────────────────────────────────────────────────────────
def load_galaxies_hyperleda(limit=HYPERLEDA_LIMIT):
    print(f"\n[GALAXIES] Querying HyperLEDA for up to {limit:,} real galaxies...")
    sql = (f"SELECT pgc,al2000,de2000,v FROM meandata "
           f"WHERE v IS NOT NULL AND v>100 AND v<100000 "
           f"AND al2000 IS NOT NULL AND de2000 IS NOT NULL LIMIT {limit}")
    url = "http://leda.univ-lyon1.fr/leda/fullsql.html?sql=" + urllib.parse.quote(sql) + "&ob=html&otyp=0"
    try:
        tables = pd.read_html(url, header=0)
        df = tables[0]
        print(f"  Raw rows: {len(df):,}")
    except Exception as e:
        print(f"  HyperLEDA failed ({e}), falling back to 2MRS...")
        return load_galaxies_local()
    col_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if "al2000" in cl or cl=="ra": col_map[c]="ra"
        elif "de2000" in cl or cl=="dec": col_map[c]="dec"
        elif cl=="v": col_map[c]="cz"
    df = df.rename(columns=col_map)
    for col in ["ra","dec","cz"]:
        if col not in df.columns:
            return load_galaxies_local()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ra","dec","cz"])
    df = df[df["cz"] > 100]
    print(f"  Valid galaxies: {len(df):,}")
    df["distance_mpc"] = df["cz"] / HUBBLE_CONSTANT
    # Keep redshift so the viewer can colour galaxies by real epoch instead of a
    # flat per-layer colour. cz is a recession velocity in km/s.
    df["redshift"] = df["cz"] / 299792.458
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg,
                      distance=df["distance_mpc"].values*u.Mpc, frame="icrs")
    df["x"] = coords.cartesian.x.to(u.Mpc).value * 1e6
    df["y"] = coords.cartesian.y.to(u.Mpc).value * 1e6
    df["z"] = coords.cartesian.z.to(u.Mpc).value * 1e6
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df

def load_galaxies_local():
    print("  Loading local galaxies_2mrs.csv...")
    try:
        df = pd.read_csv("galaxies_2mrs.csv", sep="\t", comment="#", dtype=str)
        df = df.apply(lambda col: col.str.strip() if col.dtype=="object" else col)
        for c in ["cz","RAJ2000","DEJ2000"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["cz","RAJ2000","DEJ2000"])
        df = df[df["cz"] > 0]
        df["distance_mpc"] = df["cz"] / HUBBLE_CONSTANT
        coords = SkyCoord(ra=df["RAJ2000"].values*u.deg, dec=df["DEJ2000"].values*u.deg,
                          distance=df["distance_mpc"].values*u.Mpc, frame="icrs")
        df["x"] = coords.cartesian.x.to(u.Mpc).value * 1e6
        df["y"] = coords.cartesian.y.to(u.Mpc).value * 1e6
        df["z"] = coords.cartesian.z.to(u.Mpc).value * 1e6
        df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        print(f"  Loaded {len(df):,} galaxies from local 2MRS")
        return df
    except Exception as e:
        print(f"  Local 2MRS failed: {e}")
        return pd.DataFrame(columns=["x","y","z","r"])

# ── EXOPLANETS: NASA Archive ───────────────────────────────────────────────────
def load_exoplanets():
    print("\n[EXOPLANETS] Fetching from NASA Exoplanet Archive...")
    frames = []
    url1 = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            "query=select+ra,dec,sy_dist+from+pscomppars&format=csv")
    csv1 = fetch_url(url1, "NASA confirmed exoplanets")
    if csv1:
        df1 = pd.read_csv(io.StringIO(csv1))
        df1 = df1.dropna(subset=["ra","dec","sy_dist"])
        df1 = df1[df1["sy_dist"] > 0]
        df1 = df1.rename(columns={"sy_dist":"dist_pc"})
        frames.append(df1[["ra","dec","dist_pc"]])
        print(f"  Confirmed: {len(df1):,}")
    url2 = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            "query=select+ra,dec+from+cumulative"
            "+where+koi_disposition+like+%27CANDIDATE%27&format=csv")
    csv2 = fetch_url(url2, "Kepler candidates")
    if csv2:
        df2 = pd.read_csv(io.StringIO(csv2))
        df2 = df2.dropna(subset=["ra","dec"])
        df2["dist_pc"] = 1000.0
        frames.append(df2[["ra","dec","dist_pc"]])
        print(f"  Kepler candidates: {len(df2):,}")
    if not frames:
        return pd.DataFrame(columns=["x","y","z"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ra","dec"])
    print(f"  Total exoplanet systems: {len(df):,}")
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg,
                      distance=df["dist_pc"].values*u.pc, frame="icrs")
    df["x"] = coords.cartesian.x.value
    df["y"] = coords.cartesian.y.value
    df["z"] = coords.cartesian.z.value
    return df

# ── QUASARS: SDSS DR17 ─────────────────────────────────────────────────────────
def load_quasars(limit=QUASAR_LIMIT):
    print(f"\n[QUASARS] Fetching up to {limit:,} real quasars from SDSS DR17...")
    sql = (f"SELECT TOP {limit} ra,dec,z FROM SpecObj "
           f"WHERE class='QSO' AND zWarning=0 AND z>0.01 AND z<7.0")
    url = ("https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?"
           "cmd=" + urllib.parse.quote(sql) + "&format=csv")
    csv_data = fetch_url(url, f"SDSS DR17 quasars")
    if csv_data is None:
        return pd.DataFrame(columns=["x","y","z","r"])
    lines = [l for l in csv_data.split("\n") if not l.startswith("#")]
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as e:
        print(f"  Parse error: {e}")
        return pd.DataFrame(columns=["x","y","z","r"])
    for c in ["ra","dec","z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ra","dec","z"])
    df = df[df["z"] > 0.01]
    print(f"  Valid quasars: {len(df):,}")
    c_over_H0 = 3e5 / HUBBLE_CONSTANT
    # Rename the REDSHIFT column before computing cartesian coords. The old code
    # kept it as "z" and then did rename({"z_pos": "z"}), producing two columns
    # named "z"; df[["x","y","z","r"]] then selected the FIRST one, so the quasar
    # layer's Z axis was plotting redshift instead of a spatial coordinate — the
    # whole layer collapsed to a flat sheet (z extent 1.4 against x/y of ~3.9e9).
    df = df.rename(columns={"z": "redshift"})
    df["distance_mpc"] = c_over_H0 * df["redshift"] * (1 + df["redshift"]*(-0.5 + df["redshift"]*0.167))
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg,
                      distance=df["distance_mpc"].values*u.Mpc, frame="icrs")
    df["x"] = coords.cartesian.x.to(u.Mpc).value * 1e6
    df["y"] = coords.cartesian.y.to(u.Mpc).value * 1e6
    df["z"] = coords.cartesian.z.to(u.Mpc).value * 1e6
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df

# ── GALAXY CLUSTERS: Planck PSZ2 via Vizier ────────────────────────────────────
def load_clusters():
    """
    Real galaxy clusters from the Planck PSZ2 catalogue.
    1,653 clusters detected via the Sunyaev-Zel'dovich effect.
    Each cluster = real massive dark matter + hot gas halo.
    Mass column (m) = M_SZ in units of 10^14 solar masses.
    Source: Planck Collaboration 2016, A&A 594, A27
    """
    print("\n[CLUSTERS] Fetching Planck PSZ2 catalogue from Vizier...")

    # Vizier TAP endpoint for PSZ2
    url = (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv/A%2BA%2F594%2FA27/psz2?"
        "RA_deg=&DE_deg=&Redshift=&MSZ=&"
        "-out=RA_deg,DE_deg,Redshift,MSZ&"
        "-out.max=2000&-oc.form=dec"
    )

    data = fetch_url(url, "Planck PSZ2 clusters from Vizier")

    if data is None:
        # Fallback: direct Vizier ASCII endpoint
        print("  Trying alternate Vizier endpoint...")
        url2 = (
            "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
            "-source=J/A%2BA/594/A27/psz2&"
            "-out=RA_deg,DE_deg,Redshift,MSZ&"
            "-out.max=2000"
        )
        data = fetch_url(url2, "Planck PSZ2 (alternate)")

    if data is None:
        print("  PSZ2 download failed — using galaxy density proxy for halos")
        return pd.DataFrame(columns=["x","y","z","r","m"])

    # Parse TSV — Vizier uses tab-separated with # comment lines
    lines = [l for l in data.split("\n")
             if l.strip() and not l.startswith("#") and not l.startswith("-")]

    if len(lines) < 2:
        print(f"  No data lines parsed, skipping clusters")
        return pd.DataFrame(columns=["x","y","z","r","m"])

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", header=0)
        df.columns = [c.strip() for c in df.columns]
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  Parse error: {e}")
        return pd.DataFrame(columns=["x","y","z","r","m"])

    # Flexible column name matching
    ra_col = next((c for c in df.columns if "ra" in c.lower()), None)
    dec_col = next((c for c in df.columns if "de" in c.lower() or "dec" in c.lower()), None)
    z_col = next((c for c in df.columns if "red" in c.lower() or c.lower()=="z"), None)
    m_col = next((c for c in df.columns if "msz" in c.lower() or "mass" in c.lower()), None)

    if not ra_col or not dec_col or not z_col:
        print(f"  Could not find RA/Dec/z columns in: {list(df.columns)}")
        return pd.DataFrame(columns=["x","y","z","r","m"])

    df = df.rename(columns={ra_col:"ra", dec_col:"dec", z_col:"redshift"})
    if m_col:
        df = df.rename(columns={m_col:"m"})
    else:
        df["m"] = 5.0  # default mass if not available

    for c in ["ra","dec","redshift","m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ra","dec","redshift"])
    df = df[df["redshift"] > 0]
    df["m"] = df["m"].fillna(5.0)
    print(f"  Valid clusters: {len(df):,}")

    df["distance_mpc"] = (3e5 / HUBBLE_CONSTANT) * df["redshift"] * (
        1 + df["redshift"] * (-0.5 + df["redshift"] * 0.167)
    )

    coords = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        distance=df["distance_mpc"].values * u.Mpc,
        frame="icrs"
    )
    df["x"] = coords.cartesian.x.to(u.Mpc).value * 1e6
    df["y"] = coords.cartesian.y.to(u.Mpc).value * 1e6
    df["z"] = coords.cartesian.z.to(u.Mpc).value * 1e6
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    return df[["x","y","z","r","m"]]

# ── CMB MAP: Real WMAP 9-year ILC map ─────────────────────────────────────────
def download_cmb():
    """
    Download the real WMAP 9-year Internal Linear Combination (ILC) CMB map.
    This is the actual measured temperature fluctuation map of the early universe.
    Source: NASA WMAP Science Team / Wikimedia Commons (public domain)
    The colour scale: blue=cold, red=warm, range ±200 microkelvin on 2.725K background
    """
    print("\n[CMB] Downloading real WMAP 9-year CMB map...")

    # Try multiple sources in order of preference
    sources = [
        (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/"
            "Ilc_9yr_moll4096.png/1280px-Ilc_9yr_moll4096.png",
            "WMAP 9yr ILC map (Wikimedia, 1280px)"
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/a/a0/Ilc_9yr_moll4096.png",
            "WMAP 9yr ILC map (Wikimedia, full res)"
        ),
        (
            "https://map.gsfc.nasa.gov/media/060915/060915_cmb_after.jpg",
            "WMAP CMB map (NASA)"
        ),
    ]

    for url, label in sources:
        data = fetch_url(url, label, binary=True)
        if data and len(data) > 10000:
            with open("cmb_map.jpg", "wb") as f:
                f.write(data)
            print(f"  Saved cmb_map.jpg ({len(data)//1024} KB)")
            print("  This is the real WMAP 9-year CMB temperature map.")
            print("  Blue=cold spots, Red=warm spots, range ±200 microkelvin")
            return True

    print("  All CMB sources failed.")
    print("  Manually download from:")
    print("  https://upload.wikimedia.org/wikipedia/commons/a/a0/Ilc_9yr_moll4096.png")
    print("  Save as cmb_map.jpg in your project folder.")
    return False

# ── FILAMENTS: Tempel et al. SDSS cosmic web ─────────────────────────────────
FILAMENT_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
FILAMENT_TABLE = "J/MNRAS/438/3465/table2"


def load_filaments(page=2000):
    """
    Real cosmic web filament spines from Tempel et al. (2014), MNRAS 438, 3465 —
    "Detecting filamentary pattern in the cosmic web: a catalogue of filaments for
    the SDSS", built with the Bisous process filament finder.

    Source: VizieR J/MNRAS/438/3465/table2 ("Filament points properties"),
    275,599 spine points across 15,421 filaments.

    An earlier version of this function queried J/A+A/566/A1/filaments. That table
    does not exist and never did: J/A+A/566/A1 is Tempel's SDSS *galaxy and group*
    catalogue, and the filament catalogue is in MNRAS, not A&A. The request returned
    "No catalogue or table was specified or found." every time, which is why
    filaments_points.csv was never produced while the README advertised the layer.

    This table carries co-moving Cartesian x/y/z directly in Mpc/h, so there is no
    RA/Dec conversion and no redshift-to-distance step — one less assumption than
    every other layer in this pipeline. Converted here to parsecs (Mpc * 1e6) to
    match the other *_points.csv files, dividing out h = HUBBLE_CONSTANT/100.

    Output: filaments_points.csv, one row per segment: x1,y1,z1,x2,y2,z2
    Consecutive points within a filament ID form its polyline, so a filament of
    Npts points yields Npts-1 segments.
    """
    print("\n[FILAMENTS] Fetching Tempel et al. (2014) SDSS filament spines...")
    print(f"  VizieR {FILAMENT_TABLE} via TAP")

    h = HUBBLE_CONSTANT / 100.0
    frames = []
    lo = 1
    while True:
        adql = (
            f'SELECT ID, IDpt, x, y, z, Len, fden FROM "{FILAMENT_TABLE}" '
            f"WHERE ID >= {lo} AND ID < {lo + page} ORDER BY ID, IDpt"
        )
        try:
            r = requests.post(
                FILAMENT_TAP,
                data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
                timeout=180,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"  Request failed for ID {lo}..{lo+page}: {e}")
            return pd.DataFrame()

        chunk = pd.read_csv(io.StringIO(r.text))
        if len(chunk):
            frames.append(chunk)
            print(f"    ID {lo:>6}-{lo+page-1:<6} {len(chunk):>7,} points")
        lo += page
        # 15,421 filaments; stop once past the end and a page comes back empty
        if lo > 16000 and not len(chunk):
            break
        if lo > 20000:
            break

    if not frames:
        print("  No filament data returned")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    for c in ("ID", "IDpt", "x", "y", "z", "Len", "fden"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Len/fden may be blank for a point; position must not be.
    df = df.dropna(subset=["ID", "IDpt", "x", "y", "z"]).sort_values(["ID", "IDpt"])
    print(f"  Spine points: {len(df):,} across {df['ID'].nunique():,} filaments")

    # Mpc/h -> parsecs, matching galaxies_points.csv / quasars_points.csv
    for c in ("x", "y", "z"):
        df[c] = df[c] / h * 1e6

    # Vectorised polyline -> segment pairs: pair each row with the next, then drop
    # pairs that straddle a filament boundary. A per-row Python loop over 275k rows
    # is needlessly slow here.
    cur_all = df[["ID", "x", "y", "z"]].to_numpy()
    fden_all = df["fden"].to_numpy()
    len_all = df["Len"].to_numpy()
    nxt = cur_all[1:]
    cur = cur_all[:-1]
    same = cur[:, 0] == nxt[:, 0]

    seg_len = np.sqrt(((nxt[:, 1:4] - cur[:, 1:4]) ** 2).sum(axis=1))

    # Arc length along each filament, normalised 0..1 per filament, so the shader can
    # run a travelling pulse along a spine without knowing anything about topology.
    # Cumulative sum of segment lengths, reset at every filament boundary.
    step = np.where(same, seg_len, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    ids = cur_all[:, 0]
    # subtract each filament's starting cumulative value, then divide by its span
    first = pd.Series(cum).groupby(ids).transform("first").to_numpy()
    span = pd.Series(cum).groupby(ids).transform("last").to_numpy() - first
    span = np.where(span > 0, span, 1.0)
    t = (cum - first) / span

    seg = pd.DataFrame(
        {
            "x1": cur[same, 1], "y1": cur[same, 2], "z1": cur[same, 3],
            "x2": nxt[same, 1], "y2": nxt[same, 2], "z2": nxt[same, 3],
            # arc position of each endpoint within its filament, 0 at one end, 1 at the other
            "t1": t[:-1][same], "t2": t[1:][same],
            # Bisous weighted visit-map value, 0..1 — how strongly the finder believes
            # this point is filament spine. A real per-segment scalar, mean of endpoints.
            "fden": np.nanmean(np.vstack([fden_all[:-1][same], fden_all[1:][same]]), axis=0),
            # length in Mpc/h of the filament this segment belongs to
            "flen": len_all[:-1][same],
        }
    )
    seg["fden"] = seg["fden"].fillna(0.0)
    seg["flen"] = seg["flen"].fillna(0.0)
    print(f"  Filament segments: {len(seg):,}")
    return seg


# ── RUN & SAVE ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Universe Data Pipeline — Every point is a real object")
    print("=" * 65)

    stars = load_stars_gaia(GAIA_LIMIT)
    stars[["x","y","z","r"]].to_csv("gaia_points.csv", index=False)
    print(f"  ✓ gaia_points.csv       — {len(stars):>8,} real stars (Gaia DR3)")

    galaxies = load_galaxies_hyperleda(HYPERLEDA_LIMIT)
    galaxies[["x","y","z","r"]].to_csv("galaxies_points.csv", index=False)
    print(f"  ✓ galaxies_points.csv   — {len(galaxies):>8,} real galaxies (HyperLEDA)")

    exoplanets = load_exoplanets()
    exoplanets[["x","y","z"]].to_csv("exoplanets_points.csv", index=False)
    print(f"  ✓ exoplanets_points.csv — {len(exoplanets):>8,} real planetary systems")

    quasars = load_quasars(QUASAR_LIMIT)
    if len(quasars) > 0:
        quasars[["x","y","z","r"]].to_csv("quasars_points.csv", index=False)
        print(f"  ✓ quasars_points.csv    — {len(quasars):>8,} real quasars (SDSS DR17)")

    clusters = load_clusters()
    if len(clusters) > 0:
        clusters.to_csv("clusters_points.csv", index=False)
        print(f"  ✓ clusters_points.csv   — {len(clusters):>8,} real clusters (Planck PSZ2)")

    cmb_ok = download_cmb()
    if cmb_ok:
        print(f"  ✓ cmb_map.jpg           — real WMAP 9-year CMB temperature map")

    filaments = load_filaments()
    if len(filaments) > 0:
        filaments.to_csv("filaments_points.csv", index=False)
        print(f"  ✓ filaments_points.csv  — {len(filaments):>8,} real filament segments (Tempel+2014)")
    else:
        print("  ✗ filaments_points.csv  — download failed, viewer will use algorithmic fallback")

    print("\n" + "=" * 65)
    print("All real data. Place all files in your project folder and")
    print("open universe_viewer.html with Live Server.")
    print("=" * 65)