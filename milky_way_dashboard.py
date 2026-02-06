import pandas as pd
import plotly.express as px
from astropy.coordinates import Distance
from astropy import units as u
import streamlit as st

# -----------------------------
# Title
# -----------------------------
st.title("Local Milky Way Star Map (Gaia Data) 🌌")
st.markdown("""
This dashboard shows local stars from Gaia data.
You can filter stars by distance and interactively explore their positions.
""")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("gaia_sample.csv")
    required_cols = ["l", "b", "parallax"]
    df = df.dropna(subset=required_cols)
    df["distance_pc"] = Distance(parallax=df["parallax"].values * u.mas, allow_negative=True).pc
    df = df.dropna(subset=["distance_pc"])
    return df

df = load_data()

# -----------------------------
# Sidebar controls
# -----------------------------
max_distance = int(df["distance_pc"].max())
distance_filter = st.sidebar.slider(
    "Maximum distance (parsecs)",
    min_value=0,
    max_value=max_distance,
    value=5000,
    step=100
)

filtered_df = df[df["distance_pc"] <= distance_filter]

st.sidebar.markdown(f"Displaying {len(filtered_df)} stars out of {len(df)} total")

# -----------------------------
# Plot
# -----------------------------
fig = px.scatter(
    filtered_df,
    x="l",
    y="b",
    color="distance_pc",
    labels={
        "l": "Galactic Longitude (deg)",
        "b": "Galactic Latitude (deg)",
        "distance_pc": "Distance (pc)"
    },
    title=f"Stars within {distance_filter} parsecs",
    opacity=0.6,
    hover_data={"distance_pc": True},
)

st.plotly_chart(fig, use_container_width=True)
