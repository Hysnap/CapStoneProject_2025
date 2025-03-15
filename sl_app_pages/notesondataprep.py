import streamlit as st
import sl_components.calculations as ppcalc


def notesondataprep_body():
    
    # set filters to None and filtered_df to the original dataset
    electoral_commission = "https://www.electoralcommission.org.uk/"
    wmca = "https://www.wmca.org.uk/"
    datalink = "https://www.kaggle.com/robertjacobson/uk-political-donations"
    # use markdown to create headers and sub headers
    st.write("---")
    st.write("### Notes on Data Source")
    st.write(
        "* The data was sourced from the "
        "[Electoral Commission](%s)." % electoral_commission,
        "Having been initially extracted and " "compiled by https://data.world/vizwiz.",
    )
    st.write("* The data is a snapshot of donations made to Political Parties")
    st.write("---")
    st.write("### Data Cleansing and Assumptions")
    st.write("---")
    st.write("---")
