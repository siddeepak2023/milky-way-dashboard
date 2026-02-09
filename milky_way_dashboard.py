# milky_way_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from astropy.coordinates import Distance
from astropy import units as u

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Map of the Universe",
    layout="wide"
)

st.title("🌌 Map of the Universe")

# ---------------- SCALE SELECTOR ----------------
scale = st.selectbox(
    "Choose scale",
    [
        "Earth Vicinity",
        "Solar Neighborhood",
        "Milky Way",
        "Local Group",
        "Observable Universe",
    ]
)

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("gaia_sample.csv")
except FileNotFoundError:
    st.error("gaia_sample.csv not found in repository.")
    st.stop()

required_cols = ["parallax", "l", "b"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# ---------------- CLEAN DATA ----------------
df = df.dropna(subset=required_cols)
df = df[df["parallax"] > 0]

# Distance (parsecs)
df["distance_pc"] = Distance(
    parallax=df["parallax"].values * u.mas
).pc

# Convert galactic to cartesian (disk-aligned)
l_rad = np.deg2rad(df["l"].values)
b_rad = np.deg2rad(df["b"].values)
r = df["distance_pc"].values

df["x"] = r * np.cos(b_rad) * np.cos(l_rad)
df["y"] = r * np.cos(b_rad) * np.sin(l_rad)
df["z"] = r * np.sin(b_rad)

df["radius"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

# ---------------- SCALE LOGIC ----------------
if scale == "Earth Vicinity":
    view = df[df["radius"] < 50]
    camera = dict(x=0.3, y=0.3, z=0.2)
    size = 3.5
    opacity = 0.95

elif scale == "Solar Neighborhood":
    view = df[df["radius"] < 500]
    camera = dict(x=0.7, y=0.7, z=0.4)
    size = 2.5
    opacity = 0.8

elif scale == "Milky Way":
    view = df
    camera = dict(x=1.6, y=1.6, z=0.6)
    size = 1.6
    opacity = 0.6

elif scale == "Local Group":
    view = df.sample(min(6000, len(df)))
    camera = dict(x=3, y=3, z=1.4)
    size = 1.1
    opacity = 0.35

else:  # Observable Universe
    view = df.sample(min(3000, len(df)))
    camera = dict(x=5, y=5, z=2.2)
    size = 0.9
    opacity = 0.25

# ---------------- 3D VISUALIZATION ----------------
fig = go.Figure()

# Star field
fig.add_trace(
    go.Scatter3d(
        x=view["x"],
        y=view["y"],
        z=view["z"],
        mode="markers",
        marker=dict(
            size=size,
            color=view["radius"],
            colorscale="Inferno",
            opacity=opacity,
        ),
        name="Stars",
    )
)

# Galactic core glow
fig.add_trace(
    go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        marker=dict(
            size=20,
            color="white",
            opacity=0.9,
        ),
        name="Galactic Core",
    )
)

# ---------------- LAYOUT ----------------
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=dict(eye=camera),
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor="black",
    plot_bgcolor="black",
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Scale: {scale} • Stars shown: {len(view)}")
