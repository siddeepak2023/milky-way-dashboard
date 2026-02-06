import pandas as pd
import plotly.express as px
from astropy.coordinates import Distance
from astropy import units as u

# -----------------------------
# Load Gaia tutorial data
# -----------------------------
df = pd.read_csv("gaia_sample.csv")

print("Columns in dataset:", df.columns)

# -----------------------------
# Basic cleaning
# -----------------------------
required_cols = ["l", "b", "parallax"]
df = df.dropna(subset=required_cols)

# Convert parallax (mas) to distance (pc), allow negative parallaxes
df["distance_pc"] = Distance(parallax=df["parallax"].values * u.mas, allow_negative=True).pc

# Drop any stars with NaN distance (from negative parallaxes)
df = df.dropna(subset=["distance_pc"])

# Filter to local stars for clarity
df = df[df["distance_pc"] < 5000]

# -----------------------------
# Visualization
# -----------------------------
fig = px.scatter(
    df,
    x="l",
    y="b",
    color="distance_pc",
    labels={
        "l": "Galactic Longitude (deg)",
        "b": "Galactic Latitude (deg)",
        "distance_pc": "Distance (parsecs)"
    },
    title="Local Milky Way Star Map (Gaia Data)",
    opacity=0.6
)

# Open plot in browser
fig.show(renderer="browser")
