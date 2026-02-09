import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from astropy.coordinates import Distance
from astropy import units as u

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Map of the Universe", layout="wide")
st.title("🌌 Map of the Universe")

# ---------------- CONTROLS ----------------
progress = st.slider(
    "Journey outward from Earth",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01,
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("gaia_sample.csv")
df = df.dropna(subset=["parallax", "l", "b"])
df = df[df["parallax"] > 0]

# Distance in parsecs
df["distance_pc"] = Distance(
    parallax=df["parallax"].values * u.mas
).pc

# Galactic → Cartesian
l = np.deg2rad(df["l"].values)
b = np.deg2rad(df["b"].values)
r = df["distance_pc"].values

df["x"] = r * np.cos(b) * np.cos(l)
df["y"] = r * np.cos(b) * np.sin(l)
df["z"] = r * np.sin(b)

df["radius"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

# ---------------- ZOOM CURVE ----------------
# Nonlinear curve makes motion feel cinematic
zoom_radius = 20 + (progress ** 2.8) * 20000
view = df[df["radius"] < zoom_radius]

# ---------------- VISUAL PROPERTIES ----------------
star_size = np.clip(4 - progress * 3, 0.8, 4)
star_opacity = np.clip(0.9 - progress * 0.6, 0.2, 0.9)

# ---------------- STAR FIELD ----------------
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=view["x"],
        y=view["y"],
        z=view["z"],
        mode="markers",
        marker=dict(
            size=star_size,
            color=view["radius"],
            colorscale="Inferno",
            opacity=star_opacity,
        ),
        name="Stars",
    )
)

# ---------------- GALAXY BLOBS (FAKE BUT REALISTIC) ----------------
if progress > 0.45:
    np.random.seed(7)
    blobs = 18

    gx = np.random.normal(0, zoom_radius * 0.6, blobs)
    gy = np.random.normal(0, zoom_radius * 0.6, blobs)
    gz = np.random.normal(0, zoom_radius * 0.15, blobs)

    fig.add_trace(
        go.Scatter3d(
            x=gx,
            y=gy,
            z=gz,
            mode="markers",
            marker=dict(
                size=40,
                color="rgba(180,180,255,0.15)",
            ),
            name="Galaxies",
        )
    )

# ---------------- CAMERA MOTION ----------------
camera = dict(
    eye=dict(
        x=0.6 + progress * 3.2,
        y=0.6 + progress * 3.2,
        z=0.25 + progress * 1.2,
    )
)

# ---------------- SPACE AESTHETIC ----------------
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=camera,
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor="black",
    plot_bgcolor="black",
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- STATUS ----------------
st.caption(
    f"Distance scale: {int(zoom_radius):,} parsecs • Objects rendered: {len(view)}"
)
