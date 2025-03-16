# Description: Page settings for the app
# and related functions

# Import necessary libraries
import json
import streamlit as st
from sl_utils.logger import log_function_call


@log_function_call
def load_page_settings(page):
    """Load page settings from a JSON file."""
    # page_settings filename load from st.session_state
    page_settings = st.session_state.page_settings  
    # Load the page settings
    with open(page_settings, "r") as file:
        settings = json.load(file)
    return settings.get(page, {})


# def load_tabe_settings(page):
#     """
#     Load tab settings from the config file
#     """
#     # Load the tab settings
#     tab_settings = {
#         "tab1": "Summary Statistics",
#         "tab2": "Textual Insights",
#         "tab3": "Visualizations",
#     }
#     return tab_settings

# def load_tab_contents(page):
#     """
#     Load tab contents from the config file
#     """
#     # Load the tab contents
#     tab_contents = {
#         "tab1": ["Header", "Upper", "Upper_left", "Lower_left"],
#         "tab2": ["Header", "Upper", "Upper_right", "Lower_right"],
#         "tab3": ["Header", "Upper", "Visualizations"],
#     }
#     return tab_contents

# def load_column_settings(page):
#     """
#     Load column settings from the config file
#     """
#     # Load the column settings
#     column_settings = {
#         "col1": ["Header", "Upper_left", "Lower_left"],
#         "col2": ["Header", "Upper_right", "Lower_right"],
#     }
