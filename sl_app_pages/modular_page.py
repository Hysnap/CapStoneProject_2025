import streamlit as st
import importlib
from sl_app_pages.page_configs import load_page_settings, execute_element_call
from sl_utils.logger import log_function_call, streamlit_logger as logger


@log_function_call(logger)
def display_page(functionname):
    """
    Template function to generate a Streamlit page
    with Debug Mode and Containers.
    """

    # Add Debug Mode Toggle in Sidebar
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    debug_state = st.sidebar.checkbox("🛠️ Enable Debug Mode",
                                      value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_state

    if debug_state:
        st.info("🛠️ Debug Mode is **Enabled**. Logs will be displayed below.")

    st.write(f"### 🔄 Loading `{functionname}`...")  # Debugging visual cue

    # Load Page Settings
    page_settings = load_page_settings(functionname)
    if not page_settings:
        st.error("❌ Page settings not found.")
        return

    required_elements = page_settings.get("required_elements", {})
    tab_contents = page_settings.get("tab_contents", {})

    if not isinstance(tab_contents, dict):
        st.error(f"❌ Invalid structure in `tab_contents`"
                 f" for `{functionname}`.")
        return

    # Dynamically Import Required Elements
    imported_objects = {}
    for var_name, import_path in required_elements.items():
        try:
            module_path, function_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            imported_objects[var_name] = getattr(module, function_name)
            logger.info(f"Successfully imported {function_name}"
                        f" from {module_path}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {import_path}: {e}")
            if st.session_state.debug_mode:
                st.error(f"❌ Failed to import `{import_path}`: {e}")

    # Create Streamlit Tabs
    tab1, tab2, tab3 = st.tabs([
        tab_contents.get("tab1", {}).get("Header", {}).get("content", "Tab 1"),
        tab_contents.get("tab2", {}).get("Header", {}).get("content", "Tab 2"),
        tab_contents.get("tab3", {}).get("Header", {}).get("content", "Tab 3")
    ])

    # Define Global Containers
    Header = st.container()
    Upper = st.container()

    # Process Tab 1
    with tab1:
        Tab1Header = st.container()
        Tab1Upper = st.container()
        col1, col2 = st.columns(2)
        with col1:
            tab1column1header = st.container()
            tab1Upper_left = st.container()
            tab1Lower_left = st.container()
        with col2:
            tab1column2header = st.container()
            tab1Upper_right = st.container()
            tab1Lower_right = st.container()

    # Process Tab 2
    with tab2:
        tab2Header = st.container()
        tab2Upper = st.container()
        col1, col2 = st.columns(2)
        with col1:
            tab2column1header = st.container()
            tab2Upper_left = st.container()
            tab2Lower_left = st.container()
        with col2:
            tab2column2header = st.container()
            tab2Upper_right = st.container()
            tab2Lower_right = st.container()

    # Process Tab 3
    with tab3:
        tab3Header = st.container()
        tab3Upper = st.container()
        Visualizations = st.container()

    # Execute Element Calls
    # Global Containers
    with Header:
        execute_element_call(page_settings.get("Header", {}), imported_objects)
    with Upper:
        execute_element_call(page_settings.get("Upper", {}), imported_objects)

    # Tab 1
    with Tab1Header:
        execute_element_call(tab_contents.get("tab1", {}).get("Header", {}),
                             imported_objects)
    with Tab1Upper:
        execute_element_call(tab_contents.get("tab1", {}).get("Upper", {}),
                             imported_objects)
    with tab1column1header:
        execute_element_call(tab_contents.get("tab1",
                                              {}).get("column1header", {}),
                             imported_objects)
    with tab1Upper_left:
        execute_element_call(tab_contents.get("tab1",
                                              {}).get("Upper_left", {}),
                             imported_objects)
    with tab1Lower_left:
        execute_element_call(tab_contents.get("tab1",
                                              {}).get("Lower_left", {}),
                             imported_objects)
    with tab1column2header:
        execute_element_call(tab_contents.get("tab1",
                                              {}).get("column2header", {}),
                             imported_objects)
    with tab1Upper_right:
        execute_element_call(tab_contents.get("tab1",
                                              {}).get("Upper_right", {}),
                             imported_objects)
    with tab1Lower_right:
        execute_element_call(tab_contents.get("tab1",
                                              {}).get("Lower_right", {}),
                             imported_objects)

    # Tab 2
    with tab2Header:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("Header", {}),
                             imported_objects)
    with tab2Upper:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("Upper", {}),
                             imported_objects)
    with tab2column1header:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("column1header", {}),
                             imported_objects)
    with tab2Upper_left:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("Upper_left", {}),
                             imported_objects)
    with tab2Lower_left:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("Lower_left", {}),
                             imported_objects)
    with tab2column2header:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("column2header", {}),
                             imported_objects)
    with tab2Upper_right:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("Upper_right", {}),
                             imported_objects)
    with tab2Lower_right:
        execute_element_call(tab_contents.get("tab2",
                                              {}).get("Lower_right", {}),
                             imported_objects)

    # Tab 3
    with tab3Header:
        execute_element_call(tab_contents.get("tab3",
                                              {}).get("Header", {}),
                             imported_objects)
    with tab3Upper:
        execute_element_call(tab_contents.get("tab3",
                                              {}).get("Upper", {}),
                             imported_objects)
    with Visualizations:
        execute_element_call(tab_contents.get("tab3",
                                              {}).get("Visualizations",
                                                      {}),
                             imported_objects)

    # Show Debug Logs if Debug Mode is Active
    if st.session_state.debug_mode:
        with st.expander("📝 Debug Log Output"):
            loglocation = st.session_state.get("log_fname", "app_debug.log")
            try:
                with open(loglocation, "r",
                          encoding="utf-8",
                          errors="replace") as log_file:
                    log_contents = log_file.readlines()
                    # Show the last 20 log entries
                    for line in log_contents[-20:]:
                        st.text(line.strip())
            except FileNotFoundError:
                st.error("❌ Log file not found.")

    st.success(f"✅ `{functionname}` page loaded successfully!")
# Path: sl_app_pages/page_configs.py
# end of file
