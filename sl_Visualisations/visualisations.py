import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sl_components.filters import filter_by_date
from sl_utils.logger import log_function_call, streamlit_logger


@log_function_call(streamlit_logger)
def plot_article_vs_title_polarity(target_label="Article vs Title Polarity",
                                   pageref_label="polarity_scatter"):
    """
    Plots scatter graph of article_polarity vs title_polarity
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.data_clean

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key="atpol_date_range"
    )

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df, pd.to_datetime(start_date),
                                 pd.to_datetime(end_date), "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_df,
        x="title_polarity",
        y="article_polarity",
        hue="label",  # Color by label (Real = 1, Dubious = 0)
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Article Polarity vs Title Polarity")
    ax.set_xlabel("Title Polarity")
    ax.set_ylabel("Article Polarity")
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"])

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_subject(target_label="Article Count by Subject",
                                  pageref_label="article_subject_count"):
    """
    Plots a bar chart of the count of articles by subject,
    split by Label (Real=1, Dubious=0), with color coding.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "subject" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'subject' or 'label'.")
        return

    # Aggregate count of articles per subject split by label
    article_counts = (
        df.groupby(["subject", "label"]).size().reset_index(name="count"))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.barplot(
        data=article_counts,
        x="subject",
        y="count",
        hue="label",
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Article Count by Subject (Real vs Dubious)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Count of Articles")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title="Label",
              labels=["Dubious (0)", "Real (1)"])
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_source(target_label="Article Count by Subject",
                                 pageref_label="article_subject_count"):
    """
    Plots a bar chart of the count of articles by subject,
    split by Label (Real=1, Dubious=0), with color coding.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "source_name" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'source_name' or 'label'.")
        return

    # Aggregate count of articles per subject split by label
    article_counts = df.groupby(["source_name",
                                 "label"]).size().reset_index(name="count")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.barplot(
        data=article_counts,
        x="source_name",
        y="count",
        hue="label",
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Article Count by Source (Real vs Dubious)")
    ax.set_xlabel("Source")
    ax.set_ylabel("Count of Articles")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)",
                      "Real (1)"])
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_characters(target_label="Article vs Title character",
                                     pageref_label="char_scatter"):
    """
    Plots scatter graph of text_length vs title_length
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.data_clean

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key="atp_date_range"
    )

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(4, 8))
    sns.scatterplot(
        data=filtered_df,
        x="title_length",
        y="text_length",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Character Counts Articles vs Titles")
    ax.set_xlabel("Title Character Count")
    ax.set_ylabel("Article Character Count")
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"])

    # Display visualization in Streamlit
    st.pyplot(fig)
