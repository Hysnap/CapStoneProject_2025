import streamlit as st
import pandas as pd
from sl_visualisations.map_visualisation_v2 import display_maps
from sl_utils.logger import log_function_call, streamlit_logger


@log_function_call(streamlit_logger)
def main():
    """
    Main function to display the ML Model Analysis and Visualization dashboard.
    This function performs the following steps:
    1. Sets the title of the Streamlit app.
    2. Loads data from a CSV file named 'articlesformap.csv'.
    3. Displays an error message if the data fails to load or is empty.
    4. Calls the display_maps function to visualize the data on a map.
    Returns:
        None
    """
    st.title("ML Model Analysis and Visualization")

    # Load data.articlesformap.csv
    with st.spinner('Loading data...'):
        data = pd.read_csv("data//articlesformap.csv")



    # Display Map
    display_maps()

# Path: sl_app_pages/ROD_dashboard.py
# end of ROD_dashboard.py
