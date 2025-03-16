import streamlit as st
from sl_utils.logger import log_function_call, streamlit_logger as logger


@log_function_call(logger)
def notesondataprep_body():
    # set filters to None and filtered_df to the original dataset
    Kaggle = "https://www.kaggle.com/"
    wmca = "https://www.wmca.org.uk/"
    datalink = "https://www.kaggle.com/robertjacobson/uk-political-donations"
    codeinstitute = "https://codeinstitute.net/"
    # use markdown to create headers and sub headers
    st.write("---")
    st.write("### Notes Reason Created")
    st.write("---")
    st.write("This dashboard was created as a Capstone project for"
             " the Data Analytics and AI bootcamp provided by"
             " [Code Institute](%s)." % codeinstitute)
    st.write(" The course was funded by the "
             "[West Midlands Combined Authority](%s)." % wmca)
    st.write("---")
    st.write("### Notes on Data Source")
    st.write("---")
    st.write(
        "* The data was sourced from the "
        "[Kaggle](%s)." % Kaggle,
        "The following datasets were considered:",
    )
    st.write(" The final dataset used was"
             " [Fake and Real News Dataset](%s)." % datalink)
    st.write("* The data is a snapshot of donations made to Political Parties")
    st.write("---")
    st.write("### Data Cleansing and Assumptions")
    st.write("---")
    st.write("---")
