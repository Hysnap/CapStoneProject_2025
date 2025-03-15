import plotly.express as px
import pandas as pd
import streamlit as st


def display_map(data, lat_col='latitude', lon_col='longitude', color_col='realness_score'):
    """
    Displays an interactive map with Plotly using Streamlit.

    Parameters:
    - data: Pandas DataFrame containing latitude, longitude, and score data.
    - lat_col: Column name for latitude values.
    - lon_col: Column name for longitude values.
    - color_col: Column name for color-coding data points.
    """

    if data.empty:
        st.warning("No location data available to display.")
        return

    fig = px.scatter_geo(
        data,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        hover_name="location",
        title="Geographical Distribution of Realness Scores",
        projection="natural earth",
    )

    st.plotly_chart(fig, use_container_width=True)
