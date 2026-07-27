"""Re-fetch Gaia stars KEEPING bp_rp — the real stellar colour index.

prepare_universe_data.py already selects bp_rp (load_stars_gaia), but __main__
writes only x,y,z,r, so the colour was discarded before it ever reached the
viewer. This reuses that loader and keeps the column.

    ./.venv/bin/python fetch_stars_bprp.py
"""
import prepare_universe_data as P

df = P.load_stars_gaia(P.GAIA_LIMIT)
cols = ["x", "y", "z", "r"] + (["bp_rp"] if "bp_rp" in df.columns else [])
if "bp_rp" in df.columns:
    # Keep NaN as an explicit sentinel; the packer maps it to "unknown colour"
    # rather than silently inventing a temperature.
    n_missing = int(df["bp_rp"].isna().sum())
    print(f"  bp_rp present: {len(df) - n_missing:,} / {len(df):,} ({n_missing:,} missing)")
    print(f"  bp_rp range: {df['bp_rp'].min():.3f} … {df['bp_rp'].max():.3f}")
else:
    print("  !! bp_rp absent from the Gaia response — colour cannot be real")

df[cols].to_csv("gaia_points.csv", index=False)
print(f"  ✓ gaia_points.csv — {len(df):,} stars, columns: {cols}")
