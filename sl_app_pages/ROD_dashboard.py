import streamlit as st
import pandas as pd
from sl_visualisations.map_visualisation_v2 import display_maps


def main():
    st.title("ML Model Analysis and Visualization")

    # Load data.articlesformap.csv
    dataload = st.header("Data Loading")
    data = pd.read_csv("data//articlesformap.csv")
    dataload = none

    if data is None or data.empty:
        st.error("Failed to load data. Check ETL process.")
        return

    # Display Map
    display_maps()



if __name__ == "__main__":
    main()
