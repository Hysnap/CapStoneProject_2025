import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sl_components.filters import filter_by_date
from sl_utils.logger import log_function_call, streamlit_logger
from matplotlib.ticker import FuncFormatter


@log_function_call(streamlit_logger)
def plot_article_vs_title_polarity(target_label="Article vs Title Polarity",
                                   pageref_label="polarity_scatter"):
    """
    Plots scatter graph of article_polarity vs title_polarity
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df, pd.to_datetime(start_date),
                                 pd.to_datetime(end_date), "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    my_pal = {0: "green", 1: "red"}
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=filtered_df,
        x="title_polarity",
        y="article_polarity",
        hue="label",  # Color by label (Real = 1, Dubious = 0)
        palette=my_pal,
        alpha=0.7,
        ax=ax
        )

    ax.set_title("Plot of Article Polarity vs Title Polarity",
                 fontsize=10)
    ax.set_xlabel("Title Polarity",
                  fontsize=8)
    ax.set_ylabel("Article Polarity",
                  fontsize=8)
    ax.legend(title="Label",
                labels=["Dubious (0)", "Real (1)"],
                fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_subject(target_label="Article Count by Subject",
                                  pageref_label="article_subject_count"):
    """
    Plots a bar chart of the count of articles by subject,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
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

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per subject split by label
    article_counts = df.groupby(["subject",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("subject")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="subject",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Subject (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Subject", fontsize=6)
    ax.set_ylabel(y_label, fontsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_source(target_label="Article Count by Source",
                                 pageref_label="article_source_count"):
    """
    Plots a bar chart of the count of articles by source,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
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

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per source split by label
    article_counts = df.groupby(["source_name",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("source_name")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="source_name",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Source (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Source", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(handles=handles, title="Label",
              labels=["Dubious (0)", "Real (1)"], fontsize=8)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_characters(
    target_label="Article vs Title character",
        pageref_label="char_scatter"):
    """
    Plots scatter graph of text_length vs title_length
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    my_pal = {0: "green", 1: "red"}
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_length",
        x="text_length",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
        )
    ax.set_ylim(0, 2000)

    ax.set_title("Scatter Plot of Character Counts Articles vs Titles",
                 fontsize=10)
    ax.set_ylabel("Title Character Count", fontsize=8)
    ax.set_xlabel("Article Character Count", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_media(target_label="Article Count by media",
                                pageref_label="article_media_count"):
    """
    Plots a bar chart of the count of articles by media,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "media_type" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'media_type' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per media type split by label
    article_counts = df.groupby(["media_type",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("media_type")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"

    # Sort media types by total count in descending order
    sorted_media_types = article_counts.groupby("media_type")["count"].sum().sort_values(ascending=False).index

    my_pal = {1: "green", 0: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="media_type",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7,
        order=sorted_media_types
    )

    ax.set_title("Article Count by Media (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Media Type", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"], fontsize=10)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_polarity(
    target_label="Article vs Title polarity",
        pageref_label="polarity_scatter"):
    """
    Plots scatter graph of text vs title polarity scores
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    my_pal = {0: "green", 1: "red"}
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_polarity",
        x="article_polarity",
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Polarities Articles vs Titles", fontsize=10)
    ax.set_ylabel("Title Polarity", fontsize=8)
    ax.set_xlabel("Article Polarity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_subjectivity(
    Target_label="Article vs Title subjectivity",
    pageref_label="subjectivity_scatter"):
    """
    Plots scatter graph of text vs title subjectivity scores
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                  value=False,
                  key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    my_pal = {0: "green", 1: "red"}
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="article_subjectivity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("Scatter Plot of Subjectivity Articles vs Titles",
                 fontsize=10)
    ax.set_ylabel("Title Subjectivity",
                  fontsize=8)
    ax.set_xlabel("Article Subjectivity",
                  fontsize=8)
    ax.legend(title="Label",
                labels=["Dubious (0)", "Real (1)"],
                fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_subjectivity_vs_polarity(
    target_label="Article Subjectivity vs Polarity",
    pageref_label="Article_S_V_P_scatter"):
    """
    Plots scatter graph of polarity vs article subjectivity scores
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    # Option to show/hide date slide
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
       start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    my_pal = {0: "green", 1: "red"}
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=filtered_df,
        y="article_subjectivity",
        x="article_polarity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("Scatter Plot of Article Subjectivity vs Polarity",
                 fontsize=10)
    ax.set_ylabel("Article Subjectivity",
                  fontsize=8)
    ax.set_xlabel("Article Polarity",
                  fontsize=8)
    ax.legend(title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_title_subjectivity_vs_polarity(
    target_label="Title Subjectivity vs Polarity",
    pageref_label="Title_S_V_P_scatter"):
    """
    Plots scatter graph of polarity vs title subjectivity scores
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    # Option to show/hide date slide
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
       start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    my_pal = {0: "green", 1: "red"}
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="title_polarity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("Scatter Plot of Title Subjectivity vs Polarity",
                 fontsize=10)
    ax.set_ylabel("Title Subjectivity",
                  fontsize=8)
    ax.set_xlabel("Title Polarity",
                  fontsize=8)
    ax.legend(title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_subjectivity_scat(
    Target_label="Article vs Title subjectivity",
    pageref_label="subjectivity_scatter"):
    """
    Plots scatter graph of text vs title subjectivity scores
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox(
        "Show Date Slider",
        value=False,
        key=f"{pageref_label}_slider"
    )

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Ensure `label` column exists
    if "label" not in filtered_df:
        st.error("The dataset is missing the 'label' column.")
        return

    # Convert `label` to string for correct hue mapping
    filtered_df["label"] = filtered_df["label"].astype(str)

    # Drop rows where both subjectivity values are missing
    filtered_df.dropna(subset=["title_subjectivity",
                               "article_subjectivity"],
                       how="all", inplace=True)

    # Define color palette for labels
    my_pal = {0: "green", 1: "red"}

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="article_subjectivity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    # Customize plot
    ax.set_title("Scatter Plot of Subjectivity: Articles vs Titles", fontsize=12)
    ax.set_ylabel("Title Subjectivity", fontsize=10)
    ax.set_xlabel("Article Subjectivity", fontsize=10)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_subjectivity_contrad_variations(
    target_label="Subjectivity Contradiction vs Variations",
        pageref_label="Sub_con_var_scatter"):
    """
    Plots scatter graph of subjectivity contradictions vs variations
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    my_pal = {0: "green", 1: "red"}
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="contradiction_subjectivity",
        x="subjectivity_variations",
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Subjectivity Variations and Contradictions",
                 fontsize=10)
    ax.set_ylabel("Subjectivity Contradictions", fontsize=8)
    ax.set_xlabel("Subjectivity Variations", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_polarity_contrad_variations(
    target_label="Polarity Contradiction vs Variations",
        pageref_label="Pol_con_var_scatter"):
    """
    Plots scatter graph of polarity contradictions vs variations
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    my_pal = {1: "red", 0: "green"}
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="contradiction_polarity",
        x="polarity_variations",
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Polarity Variations and Contradictions",
                 fontsize=10)
    ax.set_ylabel("Polarity\nContradictions", fontsize=8)
    ax.set_xlabel("Polarity\nVariations", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_day_label(target_label="Article Count by Day Label",
                                    pageref_label="article_day_count"):
    """
    Plots a bar chart of the count of articles by source,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "day_label" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'day_label' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per day_label split by label
    article_counts = df.groupby(["day_label",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("day_label")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="day_label",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7,
        order=article_counts["day_label"].unique()[::-1]
    )

    ax.set_title("Article Count by Day (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Day Label", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=8)
    plt.xticks(rotation=45)

    # Split y-axis

    def y_formatter(y, pos):
        if y < 10000:
            return f'{y}'
        elif 10000 <= y < 22000:
            return ''
        elif 22000 <= y < 30000:
            return f'{y - 12000}'
        elif 30000 <= y < 50000:
            return ''
        else:
            return f'{y - 22000}'

    ax.set_yscale('linear')
    ax.set_yticks([0, 10000, 22000, 30000, 50000])
    ax.get_yaxis().set_major_formatter(FuncFormatter(y_formatter))

    # Display visualization in Streamlit
    st.pyplot(fig)


# graph using data_clean to show count of Real and Dubious articles by day
@log_function_call(streamlit_logger)
def plot_article_count_by_day(target_label="Article Count by Day",
                              pageref_label="article_day_count2"):
    """
    Plots a line graph of the count of articles by day,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "date_clean" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'date_clean' or 'label'.")
        return
    df = df[df["date_clean"] >= '2015-01-01']
    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per day split by label
    article_counts = df.groupby(["date_clean",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("date_clean")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(
        data=article_counts,
        x="date_clean",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Day (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


# graph using data_clean to show count of Real and Dubious articles
# by number of locations mentioned in the article
@log_function_call(streamlit_logger)
def plot_article_count_by_location(target_label="Article Count by Location",
                                   pageref_label="article_location_count"):
    """
    Plots a bar chart of the count of articles by number of locations
    mentioned in the article,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "unique_location_count" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns:"
                 " 'unique_location_count' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per location count split by label
    article_counts = df.groupby(["unique_location_count",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("unique_location_count")
            ["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="unique_location_count",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Location Count (Real vs Dubious)",
                 fontsize=10)
    ax.set_xlabel("Location Count", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)
    plt.xticks(rotation=45)

    # Show every 5th x-axis label
    for index, label in enumerate(ax.get_xticklabels()):
        if index % 5 != 0:
            label.set_visible(False)

    # Display visualization in Streamlit
    st.pyplot(fig)


# Boxplot using data_clean to show the distribution o
# article text_count scores by label
@log_function_call(streamlit_logger)
def plot_article_text_count_distribution(
    target_label="Article Text Count Distribution",
        pageref_label="article_text_count_distribution"):
    """
    Plots a boxplot of the distribution of article text_count scores by label
    (Real=1, Dubious=0), with color coding.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "text_length" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'text_length' or 'label'.")
        return

    # Create boxplot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.boxplot(
        data=df,
        x="label",
        y="text_length",
        palette={1: "green", 0: "red"}
    )

    ax.set_title("Article Text Count Distribution by Label", fontsize=10)
    ax.set_xlabel("Label", fontsize=8)
    ax.set_ylabel("Text Count", fontsize=8)
    ax.set_xticklabels(["Dubious (0)", "Real (1)"])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    # Display visualization in Streamlit
    st.pyplot(fig)


# a set of boxplots showing all article and title
# subjectivity and polarity scores by label
@log_function_call(streamlit_logger)
def plot_polarity_subjectivity_boxplots(
    target_label="Polarity and Subjectivity Boxplots",
        pageref_label="polarity_subjectivity_boxplots"):
    """
    Plots a set of boxplots showing all article and
    title subjectivity and polarity scores by label
    (Real=1, Dubious=0), with color coding.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "title_polarity" not in df.columns or \
        "title_subjectivity" not in df.columns or \
            "article_polarity" not in df.columns or \
            "article_subjectivity" not in df.columns or \
            "label" not in df.columns:
        st.error("Dataset missing required columns: 'title_polarity',"
                 "'title_subjectivity', "
                 "'article_polarity', 'article_subjectivity' or 'label'.")
        return
    my_pal = {"0": "r", "1": "g", 0: "r", 1: "g"}
    # Create boxplot
    fig, ax = plt.subplots(3, 4, figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="label",
        y="title_polarity",
        ax=ax[0, 0],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="title_subjectivity",
        ax=ax[0, 1],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="article_polarity",
        ax=ax[1, 0],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="article_subjectivity",
        ax=ax[1, 1],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="contradiction_subjectivity",
        ax=ax[1, 2],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="subjectivity_variations",
        ax=ax[1, 3],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="contradiction_polarity",
        ax=ax[0, 2],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="polarity_variations",
        ax=ax[0, 3],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="overall_subjectivity",
        ax=ax[2, 1],
        hue="label",
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="overall_polarity",
        ax=ax[2, 2],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="text_length",
        ax=ax[2, 0],
        hue="label",
        legend=False,
        palette=my_pal,
    )
    sns.violinplot(
        data=df,
        x="label",
        y="title_length",
        ax=ax[2, 3],
        hue="label",
        legend=False,
        palette=my_pal,
    )

    ax[0, 0].set_title("Title Polarity by Label", fontsize=8)
    ax[0, 0].set_xticks([0, 1])
    ax[0, 0].set_xlabel("Label", fontsize=6)
    ax[0, 0].set_ylabel("Polarity", fontsize=6)
    ax[0, 0].tick_params(axis='x', labelsize=6)
    ax[0, 0].tick_params(axis='y', labelsize=6)
    ax[0, 0].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[0, 1].set_title("Title Subjectivity by Label", fontsize=8)
    ax[0, 1].set_xticks([0, 1])
    ax[0, 1].set_xlabel("Label", fontsize=6)
    ax[0, 1].set_ylabel("Subjectivity", fontsize=6)
    ax[0, 1].tick_params(axis='x', labelsize=6)
    ax[0, 1].tick_params(axis='y', labelsize=6)
    ax[0, 1].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[1, 0].set_title("Article Polarity by Label", fontsize=8)
    ax[1, 0].set_xticks([0, 1])
    ax[1, 0].set_xlabel("Label", fontsize=6)
    ax[1, 0].set_ylabel("Polarity", fontsize=6)
    ax[1, 0].tick_params(axis='x', labelsize=6)
    ax[1, 0].tick_params(axis='y', labelsize=6)
    ax[1, 0].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[1, 1].set_title("Article Subjectivity by Label", fontsize=8)
    ax[1, 1].set_xticks([0, 1])
    ax[1, 1].set_xlabel("Label", fontsize=6)
    ax[1, 1].set_ylabel("Subjectivity", fontsize=6)
    ax[1, 1].tick_params(axis='x', labelsize=6)
    ax[1, 1].tick_params(axis='y', labelsize=6)
    ax[1, 1].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[0, 3].set_title("Polarity Variance", fontsize=8)
    ax[0, 3].set_xticks([0, 1])
    ax[0, 3].set_xlabel("Label", fontsize=6)
    ax[0, 3].set_ylabel("Variance between Title and Text", fontsize=6)
    ax[0, 3].tick_params(axis='x', labelsize=6)
    ax[0, 3].tick_params(axis='y', labelsize=6)
    ax[0, 3].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[0, 2].set_title("Polarity Contradictions", fontsize=8)
    ax[0, 2].set_xticks([0, 1])
    ax[0, 2].set_xlabel("Label", fontsize=6)
    ax[0, 2].set_ylabel("Level of Contradictions", fontsize=6)
    ax[0, 2].tick_params(axis='x', labelsize=6)
    ax[0, 2].tick_params(axis='y', labelsize=6)
    ax[0, 2].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[1, 3].set_title("Subjectivity Variance", fontsize=8)
    ax[1, 3].set_xticks([0, 1])
    ax[1, 3].set_xlabel("Label", fontsize=6)
    ax[1, 3].set_ylabel("Variance between Title and Text", fontsize=6)
    ax[1, 3].tick_params(axis='x', labelsize=6)
    ax[1, 3].tick_params(axis='y', labelsize=6)
    ax[1, 3].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[1, 2].set_title("Subjectivity Contradictions", fontsize=8)
    ax[1, 2].set_xticks([0, 1])
    ax[1, 2].set_xlabel("Label", fontsize=6)
    ax[1, 2].set_ylabel("Level of Contradictions", fontsize=6)
    ax[1, 2].tick_params(axis='x', labelsize=6)
    ax[1, 2].tick_params(axis='y', labelsize=6)
    ax[1, 2].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[2, 1].set_title("Overall Subjectivity", fontsize=8)
    ax[2, 1].set_xticks([0, 1])
    ax[2, 1].set_xlabel("Label", fontsize=6)
    ax[2, 1].set_ylabel("Subjectivity", fontsize=6)
    ax[2, 1].tick_params(axis='x', labelsize=6)
    ax[2, 1].tick_params(axis='y', labelsize=6)
    ax[2, 1].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[2, 2].set_title("Overall Polarity", fontsize=8)
    ax[2, 2].set_xticks([0, 1])
    ax[2, 2].set_xlabel("Label", fontsize=6)
    ax[2, 2].set_ylabel("Polarity", fontsize=6)
    ax[2, 2].tick_params(axis='x', labelsize=6)
    ax[2, 2].tick_params(axis='y', labelsize=6)
    ax[2, 2].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[2, 0].set_title("Character Count Article", fontsize=8)
    ax[2, 0].set_xticks([0, 1])
    ax[2, 0].set_xlabel("Label", fontsize=6)
    ax[2, 0].set_yscale("log")
    ax[2, 0].set_ylabel("Characters", fontsize=6)
    ax[2, 0].tick_params(axis='x', labelsize=6)
    ax[2, 0].tick_params(axis='y', labelsize=6)
    ax[2, 0].set_xticklabels(["Dubious (0)", "Real (1)"])

    ax[2, 3].set_title("Character Count Title", fontsize=8)
    ax[2, 3].set_yscale("log")
    ax[2, 3].set_xticks([0, 1])
    ax[2, 3].set_xlabel("Label", fontsize=6)
    ax[2, 3].set_ylabel("Characters", fontsize=6)
    ax[2, 3].tick_params(axis='x', labelsize=6)
    ax[2, 3].tick_params(axis='y', labelsize=6)
    ax[2, 3].set_xticklabels(["Dubious (0)", "Real (1)"])

    plt.tight_layout()

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_hex_subjectivity():
    """
    Plots hexbin heatmap showing the weighted percentage of Real vs. Fake articles.
    - Uses `weights` to normalize for proportion.
    - Side histograms show volume distribution.
    """

    # Load dataset
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox(
        "Show Date Slider for Hex Plot",
        value=False,
        key="hex_slider"
    )

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="hex_date"
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data
    filtered_df = filter_by_date(df, pd.to_datetime(start_date), pd.to_datetime(end_date), "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Ensure correct data types
    filtered_df["label"] = filtered_df["label"].astype(int)

    # Handle null and near-null values
    threshold = 0.001
    filtered_df.loc[filtered_df["title_subjectivity"].abs() < threshold, "title_subjectivity"] = None
    filtered_df.loc[filtered_df["article_subjectivity"].abs() < threshold, "article_subjectivity"] = None
    filtered_df.dropna(subset=["title_subjectivity", "article_subjectivity"], how="all", inplace=True)

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    grid = sns.jointplot(
        data=filtered_df,
        x="article_subjectivity",
        y="title_subjectivity",
        kind="hex",
        cmap="coolwarm",
        gridsize=30,
        marginal_ticks=True
    )

    # Calculate weights for each hexbin (percent of Real articles)
    x = filtered_df["article_subjectivity"]
    y = filtered_df["title_subjectivity"]
    c = filtered_df["label"]  # 0 for Fake, 1 for Real

    hexbin = grid.ax_joint.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=30, cmap="coolwarm"
    )

    # Color bar
    cbar = fig.colorbar(hexbin, ax=grid.ax_joint, orientation="vertical")
    cbar.set_label("Proportion of Real Articles", fontsize=10)

    # Side histograms (volume distributions)
    sns.histplot(x, bins=30,
                 ax=grid.ax_marg_x, color="gray", kde=True)
    sns.histplot(y, bins=30,
                 ax=grid.ax_marg_y, color="gray", kde=True)

    # Titles and labels
    grid.ax_joint.set_title("Weighted Percentage of Real vs. Fake Articles",
                            fontsize=12)
    grid.ax_joint.set_xlabel("Article Subjectivity",
                             fontsize=10)
    grid.ax_joint.set_ylabel("Title Subjectivity",
                             fontsize=10)

    # Display
    st.pyplot(grid.fig)


@log_function_call(streamlit_logger)
def plot_hex_charcounts():
    """
    Plots hexbin heatmap showing the weighted percentage of Real vs. Fake articles.
    - Uses `weights` to normalize for proportion.
    - Side histograms show volume distribution.
    """

    # Load dataset
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox(
        "Show Date Slider for Hex Plot",
        value=False,
        key="hex_slider"
    )

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="hex_date"
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Ensure correct data types
    filtered_df["label"] = filtered_df["label"].astype(int)

    # Handle null and near-null values
    threshold = 10
    filtered_df.loc[filtered_df["title_length"].abs() < threshold,
                    "title_length"] = None
    filtered_df.loc[filtered_df["text_length"].abs() < threshold,
                    "text_length"] = None
    filtered_df.dropna(subset=["title_length",
                               "text_length"],
                       how="all", inplace=True)

    # Filter lengths
    filtered_df = filtered_df[(filtered_df["title_length"] < 1000) &
                              (filtered_df["text_length"] < 25000)]

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    grid = sns.jointplot(
        data=filtered_df,
        x="text_length",
        y="title_length",
        kind="hex",
        cmap="coolwarm",
        gridsize=10,
        marginal_ticks=True
    )


    # Calculate weights for each hexbin (percent of Real articles)
    x = filtered_df["text_length"]
    y = filtered_df["title_length"]
    c = filtered_df["label"]  # 0 for Fake, 1 for Real

    hexbin = grid.ax_joint.hexbin(
        x, y, C=c, reduce_C_function=np.mean,
        gridsize=50, cmap="coolwarm"
    )

    # Color bar
    cbar = fig.colorbar(hexbin,
                        ax=grid.ax_joint,
                        orientation="horizontal")
    cbar.set_label("Proportion of Real Articles",
                   fontsize=10)

    # Side histograms (volume distributions)
    sns.histplot(x, bins=25,
                 ax=grid.ax_marg_x,
                 color="gray",
                 kde=True)
    sns.histplot(y, bins=25,
                 ax=grid.ax_marg_y,
                 color="gray",
                 kde=True)


    # Titles and labels
    grid.ax_joint.set_title("Weighted Percentage of Real vs. Fake Articles",
                            fontsize=8)
    grid.ax_joint.set_xlabel("Article Size in Chars",
                             fontsize=6)
    grid.ax_joint.set_ylabel("Title Size in Chars",
                             fontsize=6)
    grid.ax_joint.set_ylim(0, 1000)
    grid.ax_joint.set_xlim(0, 25000)

    # Display
    st.pyplot(grid.fig)
