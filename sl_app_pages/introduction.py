import streamlit as st


def introduction_body():
    """
    This function displays the content of Page one.
    """
    dataset1 = (
        "https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset"
    )
    dataset2 = (
        "https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset"
    )
    codeinstitute = "https://codeinstitute.net/"
    wmca = "https://www.wmca.org.uk/"

    # format text
    st.write('### Introduction')
    st.write("* This dashboard is designed to share learnings from analysis "
             "undertaken on identifying Real or Dubious News. "
             "The data used in this analysis is sourced from the "
             f"[Fake News Corpus]({dataset1}). "
             "The analysis is based on a dataset that contains news "
             "articles from various sources. "
             "The dataset was then enhanced with further records from the "
             f"[Fake News Corpus]({dataset2}). "
             "For more details on the data, please see the "
             "Notes on Data Preparation page.")
    st.write("### Notes Reason Created")
    st.write("---")
    st.write("This dashboard was created as a Capstone project for "
             "the Data Analytics and AI bootcamp provided by "
             f"[Code Institute]({codeinstitute}). "
             "The course was funded by the "
             f"[West Midlands Combined Authority]({wmca}).")
    st.write("---")
    st.write("* The login and logout are for admin purposes only.")
    github_repo = "https://github.com/yourusername/CapStoneProject_2025-1"
    st.write(f"* You can access the code through this [GitHub repository]({github_repo})")
    st.write("---")
