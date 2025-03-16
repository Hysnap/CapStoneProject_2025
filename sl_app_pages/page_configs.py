# Description: Page settings for the app
# and related functions

# Import necessary libraries
import json
import streamlit as st
from sl_utils.logger import log_function_call, streamlit_logger as logger


@log_function_call(logger)
def load_page_settings(page_name):
    try:
        # set json file path from session_state
        page_settings_path = st.session_state.page_settings_fname

        with open(page_settings_path, "r") as f:
            config_data = json.load(f)

        if st.session_state.debug_mode:
            st.write("📜 **Loaded JSON Data:**", config_data)

        if page_name not in config_data:
            logger.warning(f"⚠️ `{page_name}` not found in JSON.")
            st.error(f"❌ Page `{page_name}` not found"
                     " in `page_settings.json`.")
            return {}

        return config_data.get(page_name, {})
    except FileNotFoundError:
        st.error("❌ JSON file not found.")
        logger.error("❌ JSON file not found.")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Decode Error: {e}")
        logger.error(f"❌ JSON Decode Error: {e}")
        return {}


@log_function_call(logger)
def execute_element_call(element_settings, imported_objects):
    """
    Executes the appropriate Streamlit function based on a flag in the JSON.
    """
    element_type = element_settings.get("type", "text")
    content = element_settings.get("content", "")

    # Interactive Debug Mode: Show debug info in Streamlit UI
    if "debug_mode" in st.session_state and st.session_state.debug_mode:
        st.write(f"🛠️ **Debug Mode Active** → `{element_type}`: `{content}`")

    logger.debug(f"Executing element: {element_type} | Content: {content}")

    if not content:
        logger.warning(f"Skipping empty element of type {element_type}.")
        if st.session_state.debug_mode:
            st.warning(f"⚠️ Skipping empty `{element_type}` element.")
        return

    if element_type == "header":
        st.header(content)
        logger.info(f"Displayed header: {content}")
    elif element_type == "text":
        st.write(content)
        logger.info(f"Displayed text: {content}")
    elif element_type == "visualization":
        try:
            module_name, function_name = content.split(".", 1)
            if module_name in imported_objects:
                function_to_call = getattr(imported_objects[module_name],
                                           function_name)
                function_to_call()
                logger.info(f"Executed visualization function: {content}")
            else:
                logger.error(f"Module `{module_name}` not found"
                             " in required elements.")
                if st.session_state.debug_mode:
                    st.error(f"❌ Module `{module_name}` not found"
                             " in required elements.")
        except Exception as e:
            logger.error(f"Execution error for `{content}`: {e}")
            if st.session_state.debug_mode:
                st.error(f"❌ Failed to execute `{content}`: {e}")
# Path: sl_app_pages/page_configs.py
# end of page_configs.py
