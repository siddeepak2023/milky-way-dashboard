"""Re-fetch SDSS quasars with a real cartesian Z and keep redshift for colour.

The old pipeline produced two columns named "z" (redshift and cartesian z), and
the CSV write picked the first — so the quasar layer rendered as a flat sheet.
"""
import prepare_universe_data as P

df = P.load_quasars(P.QUASAR_LIMIT)
print(f"  z (cartesian) extent: {df['z'].min():,.0f} … {df['z'].max():,.0f}")
print(f"  redshift range: {df['redshift'].min():.3f} … {df['redshift'].max():.3f}")
df[["x", "y", "z", "r", "redshift"]].to_csv("quasars_points.csv", index=False)
print(f"  ✓ quasars_points.csv — {len(df):,} quasars with real 3D + redshift")
