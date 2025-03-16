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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        x="title_polarity",
        y="article_polarity",
        hue="label",  # Color by label (Real = 1, Dubious = 0)
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Article Polarity vs Title Polarity", fontsize=10)
    ax.set_xlabel("Title Polarity", fontsize=8)
    ax.set_ylabel("Article Polarity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

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
    article_counts = df.groupby(["subject", "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = article_counts.groupby("subject")["count"].transform("sum")
        article_counts["percentage"] = (article_counts["count"] / total_counts) * 100
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"

    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="subject",
        y=y_value,
        hue="label",
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Article Count by Subject (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Subject", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
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
    article_counts = df.groupby(["source_name", "label"]).size().reset_index(name="count")

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

    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="source_name",
        y=y_value,
        hue="label",
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Article Count by Source (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Source", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(handles=handles, title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=8)
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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_length",
        x="text_length",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )
    ax.set_ylim(0, 2000)

    ax.set_title("Scatter Plot of Character Counts Articles vs Titles", fontsize=10)
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

    # Aggregate count of articles per subject split by label
    article_counts = df.groupby(["media_type", "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = article_counts.groupby("media_type")["count"].transform("sum")
        article_counts["percentage"] = (article_counts["count"] / total_counts) * 100
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"

    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="media_type",
        y=y_value,
        hue="label",
        palette={1: "green", 0: "red"},
        alpha=0.7
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
def plot_article_vs_title_polarity(target_label="Article vs Title polarity",
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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_polarity",
        x="article_polarity",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Polarities Articles vs Titles", fontsize=10)
    ax.set_ylabel("Title Polarity", fontsize=8)
    ax.set_xlabel("Article Polarity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_subjectivity(target_label="Article vs Title subjectivity",
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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="article_subjectivity",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Subjectivity Articles vs Titles", fontsize=10)
    ax.set_ylabel("Title Subjectivity", fontsize=8)
    ax.set_xlabel("Article Subjectivity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_title_subjectivity_vs_polarity(target_label="Title Subjectivity vs Polarity",
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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="title_polarity",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Title Subkectivity vs Polarity", fontsize=10)
    ax.set_ylabel("Title Subjectivity", fontsize=8)
    ax.set_xlabel("Title Polarity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_subjectivity_vs_polarity(target_label="Article Subjectivity vs Polarity",
                                          pageref_label="Article_S_V_P_scatter"):
    """
    Plots scatter graph of Article subjectivity vs polarity scores
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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="article_subjectivity",
        x="article_polarity",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Article Subkectivity vs Polarity", fontsize=10)
    ax.set_ylabel("Article Subjectivity", fontsize=8)
    ax.set_xlabel("Article Polarity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_subjectivity_contrad_variations(target_label="Subjectivity Contradiction vs Variations",
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
    show_slider = st.checkbox("Show Date Slider", value=False, key=f"{pageref_label}_slider")

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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="contradiction_subjectivity",
        x="subjectivity_variations",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Subjectivity Variations and Contradictions", fontsize=10)
    ax.set_ylabel("Subjectivity Contradictions", fontsize=8)
    ax.set_xlabel("Subjectivity Variations", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_polarity_contrad_variations(target_label="Polarity Contradiction vs Variations",
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

    # Date selection slider
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key=pageref_label
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
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="contradiction_polarity",
        x="polarity_variations",
        hue="label",  
        palette={1: "green", 0: "red"},
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Polarity Variations and Contradictions", fontsize=10)
    ax.set_ylabel("Polarity\nContradictions", fontsize=8)
    ax.set_xlabel("Polarity\nVariations", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)