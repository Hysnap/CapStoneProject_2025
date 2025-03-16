# import streamlit as st
# from sl_app_pages.page_configs import (load_page_settings,
#                                        execute_element_call)
# from sl_utils.logger import log_function_call
# from sl_utils.logger import streamlit_logger as logger
# import importlib

# def display_page(functionname):
#     """
#     Template function to generate a Streamlit
#     page for a specific data slice.
#     """
#     # loadpagesettings from config
#     page = functionname
#     page_settings = load_page_settings(page)
#     if not page_settings:
#         st.error("Page settings not found.")
#         return
#     # define the required elements for the page
#     required_elements = page_settings.get("required_elements", {})

#     # define the page layout using columns and containers to organize the
#     # content
#     tab_contents = page_settings.get("tab_contents", {})
#     tab1_settings = tab_contents.get("tab1", {})
#     tab2_settings = tab_contents.get("tab2", {})
#     tab3_settings = tab_contents.get("tab3", {})


#     # ✅ Dynamically import required elements
#     imported_objects = {}
#     for var_name, import_path in required_elements.items():
#         try:
#             module_path, function_name = import_path.rsplit(".", 1)
#             module = importlib.import_module(module_path)
#             imported_objects[var_name] = getattr(module, function_name)
#             logger.info(f"Successfully imported {function_name} from {module_path}")
#         except (ImportError, AttributeError) as e:
#             logger.error(f"Failed to import {import_path}: {e}")

#     # create streamlit tabs
#     tab1, tab2, tab3 = st.tabs([
#         tab1_settings.get("Header", "Tab 1"),
#         tab2_settings.get("Header", "Tab 2"),
#         tab3_settings.get("Header", "Tab 3")
#     ])
#     Header = st.container(height=100, key="Header")
#     Upper = st.container(height=200, key="Upper")
#     with tab1:
#         Tab1Header = st.container(height=100, key=f"T1header")
#         Tab1Upper = st.container(height=200, key=f"T1Upper")
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             tab1column1header = (
#                 st.container(height=100, key="T1column1header"))
#             tab1Upper_left = (
#                 st.container(height=500, key="T1Upper_left"))
#             tab1Lower_left = (
#                 st.container(height=500, key="T1Lower_left"))
#         with col2:
#             tab1column2header = (
#                 st.container(height=100, key="T1column2header"))
#             tab1Upper_right = (
#                 st.container(height=500, key="T1Upper_right"))
#             tab1Lower_right = (
#                 st.container(height=500, key="T1Lower_right"))
#     with tab2:
#         tab2Header = st.container(height=100, key=f"T2header")
#         tab2Upper = st.container(height=200, key=f"T2Upper")
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             tab2column1header = (
#                 st.container(height=100, key="T2column1header"))
#             tab2Upper_left = (
#                 st.container(height=500, key="T2Upper_left"))
#             tab2Lower_left = (
#                 st.container(height=500, key="T2Lower_left"))
#         with col2:
#             tab2column2header = (
#                 st.container(height=100, key="T2column2header"))
#             tab2Upper_right = (
#                 st.container(height=500, key="T2Upper_right"))
#             tab2Lower_right = (
#                 st.container(height=500, key="T2Lower_right"))
#     with tab3:
#         tab3Header = (
#             st.container(height=100, key=f"T3header"))
#         tab3Upper = (
#             st.container(height=200, key=f"T3Upper"))
#         Visualizations = (
#             st.container(height=500, key=f"T3Visualizations"))

# # setting container contents based on the page settings
#     # Page contents
#     with Header:
#         st.title(page_settings.get("Header", ""))
#     with Upper:
#         st.write(page_settings.get("Upper", ""))
#     # Tab 1
#     with Tab1Header:
#         st.title(tab1_settings.get("Header", ""))
#     with Tab1Upper:
#         st.write(tab1_settings.get("Upper", ""))
#     with tab1column1header:
#         st.write(tab1_settings.get("column1header", ""))
#     with tab1Upper_left:
#         st.write(tab1_settings.get("Upper_left", ""))
#     with tab1Lower_left:
#         st.write(tab1_settings.get("Lower_left", ""))
#     with tab1column2header:
#         st.write(tab1_settings.get("column2header", ""))
#     with tab1Upper_right:
#         st.write(tab1_settings.get("Upper_right", ""))
#     with tab1Lower_right:
#         st.write(tab1_settings.get("Lower_right", ""))
#     # Tab 2
#     with tab2Header:
#         st.title(tab2_settings.get("Header", ""))
#     with tab2Upper:
#         st.write(tab2_settings.get("Upper", ""))
#     with tab2column1header:
#         st.write(tab2_settings.get("column1header", ""))
#     with tab2Upper_left:
#         st.write(tab2_settings.get("Upper_left", ""))
#     with tab2Lower_left:
#         st.write(tab2_settings.get("Lower_left", ""))
#     with tab2column2header:
#         st.write(tab2_settings.get("column2header", ""))
#     with tab2Upper_right:
#         st.write(tab2_settings.get("Upper_right", ""))
#     with tab2Lower_right:
#         st.write(tab2_settings.get("Lower_right", ""))
#     # Tab 3
#     with tab3Header:
#         st.title(tab3_settings.get("Header", ""))
#     with tab3Upper:
#         st.write(tab3_settings.get("Upper", ""))
#     with Visualizations:
#         (tab3_settings.get("Visualizations", ""))

#     # Display the page
#     st.write("Page generated successfully.")
#     return
# # End of display_page
# # Path: sl_app_pages/page_configs.py
# # End of modular_page.py


# @log_function_call(logger)
# def display_page(functionname):
#     """Template function to generate a Streamlit page with Debug Mode."""

#     # ✅ Add Debug Mode Toggle in Sidebar
#     if "debug_mode" not in st.session_state:
#         st.session_state.debug_mode = False

#     debug_state = st.sidebar.checkbox("🛠️ Enable Debug Mode", value=st.session_state.debug_mode)
#     st.session_state.debug_mode = debug_state

#     if debug_state:
#         st.info("🛠️ Debug Mode is **Enabled**. Logs will be displayed below.")

#     st.write(f"### 🔄 Loading `{functionname}`...")  # Debugging visual cue
#     page_settings = load_page_settings(functionname)
#     if not page_settings:
#         st.error("❌ Page settings not found.")
#         return

#     required_elements = page_settings.get("required_elements", {})
#     tab_contents = page_settings.get("tab_contents", {})

#     if not isinstance(tab_contents, dict):
#         st.error(f"❌ Invalid structure in `tab_contents` for `{functionname}`.")
#         return

#     # ✅ Dynamically import required elements
#     imported_objects = {}
#     for var_name, import_path in required_elements.items():
#         try:
#             module_path, function_name = import_path.rsplit(".", 1)
#             module = importlib.import_module(module_path)
#             imported_objects[var_name] = getattr(module, function_name)
#             logger.info(f"✅ Successfully imported {function_name} from {module_path}")
#         except (ImportError, AttributeError) as e:
#             logger.error(f"❌ Failed to import {import_path}: {e}")
#             if st.session_state.debug_mode:
#                 st.error(f"❌ Failed to import `{import_path}`: {e}")

#     # ✅ Create Streamlit tabs
#     tab_names = [
#         tab_contents.get("tab1", {}).get("Header", {}).get("content", "Tab 1"),
#         tab_contents.get("tab2", {}).get("Header", {}).get("content", "Tab 2"),
#         tab_contents.get("tab3", {}).get("Header", {}).get("content", "Tab 3")
#     ]
#     tab1, tab2, tab3 = st.tabs(tab_names)

#     # ✅ Process each tab dynamically
#     for tab, tab_key in zip([tab1, tab2, tab3], ["tab1", "tab2", "tab3"]):
#         with tab:
#             tab_settings = tab_contents.get(tab_key, {})

#             if not isinstance(tab_settings, dict):
#                 st.warning(f"⚠️ Tab `{tab_key}` is incorrectly structured. Skipping...")
#                 logger.warning(f"⚠️ Tab `{tab_key}` is incorrectly structured. Skipping...")
#                 continue

#             st.write(f"#### 🔎 Processing `{tab_key}`...")
#             execute_element_call(tab_settings.get("Header", {}), imported_objects)
#             execute_element_call(tab_settings.get("Upper", {}), imported_objects)
#             execute_element_call(tab_settings.get("Visualizations", {}), imported_objects)

#     # ✅ Show Debug Logs if Debug Mode is Active
#     if st.session_state.debug_mode:
#         with st.expander("📝 Debug Log Output"):
#             loglocation = st.session_state.log_fname
#             with open(loglocation, "r") as log_file:
#                 log_contents = log_file.readlines()
#                 for line in log_contents[-20:]:  # Show the last 20 log entries
#                     st.text(line.strip())

#     st.success(f"✅ `{functionname}` page loaded successfully!")


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

    # ✅ Add Debug Mode Toggle in Sidebar
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    debug_state = st.sidebar.checkbox("🛠️ Enable Debug Mode",
                                      value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_state

    if debug_state:
        st.info("🛠️ Debug Mode is **Enabled**. Logs will be displayed below.")

    st.write(f"### 🔄 Loading `{functionname}`...")  # Debugging visual cue

    # ✅ Load Page Settings
    page_settings = load_page_settings(functionname)
    if not page_settings:
        st.error("❌ Page settings not found.")
        return

    required_elements = page_settings.get("required_elements", {})
    tab_contents = page_settings.get("tab_contents", {})

    if not isinstance(tab_contents, dict):
        st.error(f"❌ Invalid structure in `tab_contents` for `{functionname}`.")
        return

    # ✅ Dynamically Import Required Elements
    imported_objects = {}
    for var_name, import_path in required_elements.items():
        try:
            module_path, function_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            imported_objects[var_name] = getattr(module, function_name)
            logger.info(f"✅ Successfully imported {function_name} from {module_path}")
        except (ImportError, AttributeError) as e:
            logger.error(f"❌ Failed to import {import_path}: {e}")
            if st.session_state.debug_mode:
                st.error(f"❌ Failed to import `{import_path}`: {e}")

    # ✅ Create Streamlit Tabs
    tab1, tab2, tab3 = st.tabs([
        tab_contents.get("tab1", {}).get("Header", {}).get("content", "Tab 1"),
        tab_contents.get("tab2", {}).get("Header", {}).get("content", "Tab 2"),
        tab_contents.get("tab3", {}).get("Header", {}).get("content", "Tab 3")
    ])

    # ✅ Define Global Containers
    Header = st.container()
    Upper = st.container()

    # ✅ Process Tab 1
    with tab1:
        Tab1Header = st.container()
        Tab1Upper = st.container()
        col1, col2 = st.columns([1, 2])
        with col1:
            tab1column1header = st.container()
            tab1Upper_left = st.container()
            tab1Lower_left = st.container()
        with col2:
            tab1column2header = st.container()
            tab1Upper_right = st.container()
            tab1Lower_right = st.container()

    # ✅ Process Tab 2
    with tab2:
        tab2Header = st.container()
        tab2Upper = st.container()
        col1, col2 = st.columns([1, 2])
        with col1:
            tab2column1header = st.container()
            tab2Upper_left = st.container()
            tab2Lower_left = st.container()
        with col2:
            tab2column2header = st.container()
            tab2Upper_right = st.container()
            tab2Lower_right = st.container()

    # ✅ Process Tab 3
    with tab3:
        tab3Header = st.container()
        tab3Upper = st.container()
        Visualizations = st.container()

    # ✅ Execute Element Calls
    # 🔹 Global Containers
    with Header:
        execute_element_call(page_settings.get("Header", {}), imported_objects)
    with Upper:
        execute_element_call(page_settings.get("Upper", {}), imported_objects)

    # 🔹 Tab 1
    with Tab1Header:
        execute_element_call(tab_contents.get("tab1", {}).get("Header", {}),
                             imported_objects)
    with Tab1Upper:
        execute_element_call(tab_contents.get("tab1", {}).get("Upper", {}),
                             imported_objects)
    with tab1column1header:
        execute_element_call(tab_contents.get("tab1", {}).get("column1header", {}),
                             imported_objects)
    with tab1Upper_left:
        execute_element_call(tab_contents.get("tab1", {}).get("Upper_left", {}),
                             imported_objects)
    with tab1Lower_left:
        execute_element_call(tab_contents.get("tab1", {}).get("Lower_left", {}),
                             imported_objects)
    with tab1column2header:
        execute_element_call(tab_contents.get("tab1", {}).get("column2header", {}),
                             imported_objects)
    with tab1Upper_right:
        execute_element_call(tab_contents.get("tab1", {}).get("Upper_right", {}),
                             imported_objects)
    with tab1Lower_right:
        execute_element_call(tab_contents.get("tab1", {}).get("Lower_right", {}),
                             imported_objects)

    # 🔹 Tab 2
    with tab2Header:
        execute_element_call(tab_contents.get("tab2", {}).get("Header", {}),
                             imported_objects)
    with tab2Upper:
        execute_element_call(tab_contents.get("tab2", {}).get("Upper", {}),
                             imported_objects)
    with tab2column1header:
        execute_element_call(tab_contents.get("tab2", {}).get("column1header", {}),
                             imported_objects)
    with tab2Upper_left:
        execute_element_call(tab_contents.get("tab2", {}).get("Upper_left", {}),
                             imported_objects)
    with tab2Lower_left:
        execute_element_call(tab_contents.get("tab2", {}).get("Lower_left", {}),
                             imported_objects)
    with tab2column2header:
        execute_element_call(tab_contents.get("tab2", {}).get("column2header", {}),
                             imported_objects)
    with tab2Upper_right:
        execute_element_call(tab_contents.get("tab2", {}).get("Upper_right", {}),
                             imported_objects)
    with tab2Lower_right:
        execute_element_call(tab_contents.get("tab2", {}).get("Lower_right", {}),
                             imported_objects)

    # 🔹 Tab 3
    with tab3Header:
        execute_element_call(tab_contents.get("tab3", {}).get("Header", {}),
                             imported_objects)
    with tab3Upper:
        execute_element_call(tab_contents.get("tab3", {}).get("Upper", {}),
                             imported_objects)
    with Visualizations:
        execute_element_call(tab_contents.get("tab3", {}).get("Visualizations",
                                                              {}),
                             imported_objects)

    # ✅ Show Debug Logs if Debug Mode is Active
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