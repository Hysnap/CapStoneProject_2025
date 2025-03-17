# Description: Page settings for the app
# and related functions

# Import necessary libraries
import json
import streamlit as st
from sl_utils.logger import log_function_call, streamlit_logger as logger
import importlib

@log_function_call(logger)
def load_page_settings(page_name):
    try:
        # set json file path from session_state
        page_settings_path = st.session_state.page_settings_fname

        with open(page_settings_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if st.session_state.debug_mode:
            st.write("📜 **Loaded JSON Data:**", config_data)

        if page_name not in config_data:
            logger.warning(f"`{page_name}` not found in JSON.")
            st.error(f"❌ Page `{page_name}` not found"
                     " in `page_settings.json`.")
            return {}

        return config_data.get(page_name, {})
    except FileNotFoundError:
        st.error("❌ JSON file not found.")
        logger.error("JSON file not found.")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Decode Error: {e}")
        logger.error(f"JSON Decode Error: {e}")
        return {}


@log_function_call(logger)
def execute_element_call(element_settings, imported_objects):
    """
    Executes the appropriate Streamlit function based on a flag in the JSON.
    """
    element_type = element_settings.get("type", "text")
    content = element_settings.get("content", "")

    # Interactive Debug Mode: Show debug info in Streamlit UI
    if st.session_state.get("debug_mode", False):
        st.write(f"🛠️ **Debug Mode Active** → `{element_type}`: `{content}`")

    logger.debug(f"Executing element: {element_type} | Content: {content}")

    if not content:
        logger.warning(f"Skipping empty element of type {element_type}.")
        if st.session_state.get("debug_mode", False):
            st.warning(f"⚠️ Skipping empty `{element_type}` element.")
        return

    if element_type == "header":
        st.header(content)
        logger.info(f"Displayed header: {content}")

    elif element_type == "subheader":
        st.subheader(content)
        logger.info(f"Displayed subheader: {content}")

    elif element_type == "function":
        try:
            function_to_call = getattr(imported_objects, content)
            function_to_call()
            logger.info(f"Successfully executed function: {content}")
        except AttributeError:
            logger.error(f"Function `{content}` not found in required elements.")
            if st.session_state.get("debug_mode", False):
                st.error(f"❌ Function `{content}` not found.")

    elif element_type == "markdown":
        st.markdown(content)
        logger.info(f"Displayed markdown: {content}")

    elif element_type == "text":
        st.write(content)
        logger.info(f"Displayed text: {content}")

    elif element_type == "visualization":
        try:
            execute_visualization(content)  # Use the centralized function
            logger.info(f"Successfully executed visualization: {content}")
        except Exception as e:
            logger.error(f"Execution error for visualization `{content}`: {e}")
            if st.session_state.get("debug_mode", False):
                st.error(f"❌ Failed to execute `{content}`: {e}")

@log_function_call(logger)
def execute_visualization(content):
    """
    Dynamically executes a visualization function based on the given 
    `references.folder.module_name.function_name` format.

    Example:
        - "components.visualizations.plot_article_vs_title_polarity"
        - "modules.graphs.display_maps"

    Parameters:
        content (str): Full function path in the format 
                      "folder.module_name.function_name"

    Returns:
        None (Executes the visualization function in Streamlit)
    """
    try:
        # Extract folder, module, and function
        parts = content.split(".", 2)
        if len(parts) != 3:
            raise ValueError("Invalid format! Use folder.module_name.function_name")

        folder, module_name, function_name = parts
        full_module_path = f"{folder}.{module_name}"

        logger.info(f"🔍 Attempting to execute visualization: {full_module_path}.{function_name}")

        # Try to access the function if the module is already imported
        if module_name in globals():
            function_to_call = getattr(globals()[module_name], function_name)
            function_to_call()  # Execute
            logger.info(f"Successfully executed visualization from global context: {content}")
            return

        # Import module dynamically
        try:
            module = importlib.import_module(full_module_path)
            function_to_call = getattr(module, function_name)
            function_to_call()  # Execute visualization
            logger.info(f"Successfully executed visualization: {content}")
        except ImportError as imp_err:
            logger.error(f"ImportError: Could not load `{full_module_path}`. Error: {imp_err}")
            if st.session_state.get("debug_mode", False):
                st.error(f"❌ ImportError: {full_module_path} not found. {imp_err}")
        except AttributeError:
            logger.error(f"Function `{function_name}` not found in `{full_module_path}`.")
            if st.session_state.get("debug_mode", False):
                st.error(f"❌ Function `{function_name}` not found in `{full_module_path}`.")

    except Exception as e:
        logger.error(f"Execution error for `{content}`: {e}")
        if st.session_state.get("debug_mode", False):
            st.error(f"❌ Failed to execute `{content}`: {e}")

# Path: sl_app_pages/page_configs.py
# end of page_configs.py
