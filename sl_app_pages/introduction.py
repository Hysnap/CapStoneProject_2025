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

    # format text
    st.write('### Introduction')
    st.write("* This dashboard is designed to share learnings from analysis "
             "undertaken on identifying Real or Dubious News")
    st.write("* The data used in this analysis is sourced from the "
             f"[Fake News Corpus]({dataset1}).")
    st.write("* The analysis is based on a dataset that contains news"
             " articles from various sources.")
    st.write("* The dataset was then enhanced with further records from the "
             f"[Fake News Corpus]({dataset2}.")
    st.write("* For more details on the data, please see the "
             "Notes on Data Preparation page.")
    st.write("* This dashboard was created as a Capstone project"
             " for the Data Analytics and AI bootcamp provided"
             " by Code Institute.  The course was funded by the West "
             "Midlands Combined Authority")
    st.write("---")
    st.write("The analysis is divided into the following sections:")
    st.write("1. Objective")
    st.write("2. Data Evaluation")
    st.write("3. Exploratory Data Analysis")
    st.write("### Objective")
    st.write("* The objective of this analysis is to identify patterns that "
             "can help distinguish between real and fake news.")
    st.write("* The analysis includes the following steps:")
    st.write("    * Data Preparation")
    st.write("    * Exploratory Data Analysis")
    st.write("    * Feature Engineering")
    st.write("    * Model Building")
    st.write("    * Model Evaluation")
    st.write("    * Model Deployment")
    st.write("---")
    st.write("* The login and logout are for admin purposes only.")
    st.write("* You should be able to access the code through the github link"
             " at the top of the page.")
    st.write("---")
