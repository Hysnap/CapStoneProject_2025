import streamlit as st
from sl_utils.logger import log_function_call, streamlit_logger as logger


@log_function_call(logger)
def notesondataprep_body():
 
    # set filters to None and filtered_df to the original dataset
    Kaggle = "https://www.kaggle.com/"
    wmca = "https://www.wmca.org.uk/"
    datalink = "https://www.kaggle.com/robertjacobson/uk-political-donations"
    # use markdown to create headers and sub headers
    st.write("---")
    st.write("### Notes on Data Source")
    st.write(
        "* The data was sourced from the "
        "[Kaggle](%s)." % Kaggle,
        "The following datasets were considered:",
    )
    st.write("* The data is a snapshot of donations made to Political Parties")
    st.write("---")
    st.write("### Data Cleansing and Assumptions")
    st.write("---")
    st.write("---")
