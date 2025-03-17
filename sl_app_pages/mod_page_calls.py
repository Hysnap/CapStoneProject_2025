import streamlit as st
from sl_reference_files.text_management import check_password
from sl_utils.logger import streamlit_logger as logger  # Import the logger
from sl_utils.logger import log_function_call  # Import decorator
from sl_app_pages.modular_page import display_page
from sl_app_pages.datacleanliness import visualize_data_cleanliness
from sl_visualisations.map_visualisation import display_maps
# from sl_app_pages.ML_page import load_model


# login
@log_function_call(logger)
def loginpage():
    # page_texts = load_page_text("login")

    if "is_admin" not in st.session_state:
        st.session_state.security["is_admin"] = False

    if not st.session_state.security["is_admin"]:
        st.subheader("Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_password(username, password):
                st.session_state.security["is_admin"] = True
                st.success("Login successful!")
                logger.info("User is not logged in as admin.")
            else:
                st.error("Invalid username or password.")
                st.success("You are not logged in as admin.")
                logger.info("User is not logged in as admin.")
    else:
        st.warning("You are already logged in as admin.")


# logout
@log_function_call(logger)
def logoutpage():
    # page_texts = load_page_text("logout")

    if "is_admin" not in st.session_state:
        st.session_state.security["is_admin"] = False

    # Logout button
    if st.session_state.security["is_admin"]:
        if st.button("Logout"):
            st.session_state.security["is_admin"] = False
            st.success("Logout successful!")


# modular page calls
@log_function_call(logger)
def mp1_intro():
    display_page("Objective")


@log_function_call(logger)
def mp2_dataex():
    display_page("Data Exploration")


@log_function_call(logger)
def mp3_datapre():
    display_page("Data Preprocessing")


@log_function_call(logger)
def mp4_datapre():
    display_page("Data Date Analysis")


@log_function_call(logger)
def interactive_map():
    display_maps()


@log_function_call(logger)
def dcl_fakecsv():
    datafile = "fake_news_sources_fname"
    visualize_data_cleanliness(datafile)


@log_function_call(logger)
def dcl_truecsv():
    datafile = "true_news_sources_fname"
    visualize_data_cleanliness(datafile)


@log_function_call(logger)
def dcl_combinedcsv():
    datafile = "combined_misinfo_fname"
    visualize_data_cleanliness(datafile)


@log_function_call(logger)
def machinelearning():
    # load_model()
    return

# End of mod_page_calls.py
