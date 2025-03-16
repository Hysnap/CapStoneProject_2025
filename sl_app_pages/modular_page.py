import streamlit as st
from sl_app_pages.page_configs import load_page_settings
from sl_utils.logger import log_function_call  # Import decorator


@log_function_call
def display_page(functionname):
    """
    Template function to generate a Streamlit
    page for a specific data slice.
    """
    # loadpagesettings from config
    page = functionname
    page_settings = load_page_settings(page)
    if not page_settings:
        st.error("Page settings not found.")
        return
    # Load tab settings
    # load data

    # define the page layout using columns and containers to organize the
    # content
    tab1, tab2, tab3 = st.tabs([page_settings["tab1"],
                                page_settings["tab2"],
                                page_settings["tab3"]])
    with tab1:
        Header = st.container(height=100, key=f"{tab1}header")
        Upper = st.container(height=200, key=f"{tab1}Upper")
        col1, col2 = st.columns([1, 2])
        with col1:
            tab1column1header = (
                st.container(height=100, key="T1column1header"))
            tab1Upper_left = (
                st.container(height=500, key="T1Upper_left"))
            tab1Lower_left = (
                st.container(height=500, key="T1Lower_left"))
        with col2:
            tab1column2header = (
                st.container(height=100, key="T1column2header"))
            tab1Upper_right = (
                st.container(height=500, key="T1Upper_right"))
            tab1Lower_right = (
                st.container(height=500, key="T1Lower_right"))
    with tab2:
        tab2Header = st.container(height=100, key=f"{tab2}header")
        tab2Upper = st.container(height=200, key=f"{tab2}Upper")
        col1, col2 = st.columns([1, 2])
        with col1:
            tab2column1header = (
                st.container(height=100, key="T2column1header"))
            tab2Upper_left = (
                st.container(height=500, key="T2Upper_left"))
            tab2Lower_left = (
                st.container(height=500, key="T2Lower_left"))
        with col2:
            tab2column2header = (
                st.container(height=100, key="T2column2header"))
            tab2Upper_right = (
                st.container(height=500, key="T2Upper_right"))
            tab2Lower_right = (
                st.container(height=500, key="T2Lower_right"))
    with tab3:
        tab3Header = (
            st.container(height=100, key=f"{tab3}header"))
        tab3Upper = (
            st.container(height=200, key=f"{tab3}Upper"))
        Visualizations = (
            st.container(height=500, key=f"{tab3}Visualizations"))

# setting container contents based on the page settings

    # Tab 1
    with Header:
        st.title(page_settings["header"])
    with Upper:
        st.write(page_settings["upper"])
    with tab1column1header:
        st.write(page_settings["column1header"])
    with tab1Upper_left:
        st.write(page_settings["upper_left"])
    with tab1Lower_left:
        st.write(page_settings["lower_left"])
    with tab1column2header:
        st.write(page_settings["column2header"])
    with tab1Upper_right:
        st.write(page_settings["upper_right"])
    with tab1Lower_right:
        st.write(page_settings["lower_right"])
    # Tab 2
    with tab2Header:
        st.title(page_settings["header"])
    with tab2Upper:
        st.write(page_settings["upper"])
    with tab2column1header:
        st.write(page_settings["column1header"])
    with tab2Upper_left:
        st.write(page_settings["upper_left"])
    with tab2Lower_left:
        st.write(page_settings["lower_left"])
    with tab2column2header:
        st.write(page_settings["column2header"])
    with tab2Upper_right:
        st.write(page_settings["upper_right"])
    with tab2Lower_right:
        st.write(page_settings["lower_right"])
    # Tab 3
    with tab3Header:
        st.title(page_settings["header"])
    with tab3Upper:
        st.write(page_settings["upper"])
    with Visualizations:
        st.write(page_settings["visualizations"])
    # Display the page
    st.write("Page generated successfully.")
    return
# End of display_page
# End of modular_page.py
