# milky_way_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u

# MUST BE FIRST
st.set_page_config(
    page_title="Map of the Universe",
    layout="wide",
    page_icon="🌌"
)

st.title("Map of the Universe")

# ----------------------
# SCALE SELECTOR
# ----------------------
scale = st.selectbox(
    "Choose scale",
    [
        "Earth vicinity",
        "Solar neighborhood",
        "Milky Way",
        "Local Group",
        "Observable Universe"
    ]
)

# ----------------------
# LOAD DATA
# ----------------------
try:
    df = pd.read_csv("gaia_sample.csv")
except FileNotFoundError:
    st.error("gaia_sample.csv not found in repo")
    st.stop()

required_cols = ["parallax", "l", "b"]
df = df.dropna(subset=required_cols)
df = df[df["parallax"] > 0]

# ----------------------
# CONVERT TO REAL 3D SPACE
# ----------------------
coords = SkyCoord(
    l=df["l"].values * u.deg,
    b=df["b"].values * u.deg,
    distance=Distance(
        parallax=df["parallax"].values * u.mas,
        allow_negative=True
    ),
    frame="galactic"
)


df["x"] = coords.cartesian.x.value
df["y"] = coords.cartesian.y.value
df["z"] = coords.cartesian.z.value
df["r"] = np.sqrt(df.x**2 + df.y**2 + df.z**2)

# ----------------------
# SCALE LOGIC
# ----------------------
if scale == "Earth vicinity":
    df = df[df["r"] < 50]
    camera_eye = dict(x=0.5, y=0.5, z=0.5)

elif scale == "Solar neighborhood":
    df = df[df["r"] < 500]
    camera_eye = dict(x=1, y=1, z=0.6)

elif scale == "Milky Way":
    df = df[df["r"] < 5000]
    camera_eye = dict(x=1.8, y=1.8, z=0.6)

elif scale == "Local Group":
    camera_eye = dict(x=3, y=3, z=1)

elif scale == "Observable Universe":
    camera_eye = dict(x=6, y=6, z=2)

# Star appearance
df["size"] = np.clip(4 / df["r"], 0.3, 3)

# ----------------------
# 3D STAR FIELD
# ----------------------
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    size="size",
    color="r",
    color_continuous_scale="Inferno",
    opacity=0.6
)

# ----------------------
# ADD SYNTHETIC GALAXIES (FOR LARGE SCALES)
# ----------------------
if scale in ["Local Group", "Observable Universe"]:
    galaxies = np.random.normal(scale=30000, size=(3000, 3))
    fig.add_scatter3d(
        x=galaxies[:, 0],
        y=galaxies[:, 1],
        z=galaxies[:, 2],
        mode="markers",
        marker=dict(size=1, opacity=0.15, color="white"),
        name="Galaxies"
    )

# ----------------------
# SPACE AESTHETICS
# ----------------------
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="black"
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    margin=dict(l=0, r=0, t=0, b=0),
    scene_camera=dict(eye=camera_eye)
)

fig.update_traces(marker=dict(line=dict(width=0)))

# ----------------------
# DISPLAY
# ----------------------
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Stars rendered: {len(df):,}")
