import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sl_utils.logger import log_function_call, streamlit_logger as logger

@log_function_call(logger)
def visualize_data_cleanliness(datafile):
    st.title("🔍 Data Cleanliness Dashboard")

    # Load dataset from session state
    df_name = st.session_state.get(datafile, None)
    try:
        with st.spinner('Loading dataset...'):
            df = pd.read_csv(df_name)
        if df_name is None:
            st.error("No dataset loaded. Please upload a dataset.")
            return
        else:
            st.write("### Data Overview")
            st.write(df.info())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

    df["text_length"] = df["text"].apply(len)
    df["title_length"] = df["title"].apply(len)
    df["title_word_count"] = df["title"].apply(lambda x: len(x.split()))
    df["text_word_count"] = df["text"].apply(lambda x: len(x.split()))

    #if there is a date column, convert it to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    col1, col2 = st.columns(2)
    with col1: # Display first 5 rows
        st.write("### First 5 Rows")
        st.write(df.head())

        # Missing values heatmap
        st.write("### Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(df.isnull(),
                    cmap="viridis",
                    cbar=False,
                    yticklabels=False,
                    ax=ax)
        st.pyplot(fig)

        # Missing values count
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            st.write("### Missing Values per Column")
            fig, ax = plt.subplots(figsize=(4, 2))
            missing_values.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45)
            plt.title("Missing Values per Column")
            st.pyplot(fig)
        else:
            st.success("No missing values detected!")

        # Correlation heatmap
        st.write("### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # Data type analysis
        st.write("### Data Types Count")
        fig, ax = plt.subplots(figsize=(4, 2))
        df.dtypes.value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)
    with col2:
        # Duplicate records
        duplicate_count = df.duplicated().sum()
        st.write(f"### Duplicate Records: {duplicate_count}")

        # Summary statistics
        st.write("### Summary Statistics")
        st.write(df.describe())

        # Outlier detection using boxplots
        st.write("### Outlier Detection (Boxplots)")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            for col in numerical_cols:
                fig, ax = plt.subplots(figsize=(4, 2))
                sns.boxplot(y=df[col], ax=ax)
                plt.title(f"Boxplot for {col}")
                st.pyplot(fig)
        else:
            st.warning("No numerical columns available for boxplot analysis.")

        # Data distribution
        st.write("### Data Distribution (Histograms)")
        fig, ax = plt.subplots(figsize=(4, 2))
        df.hist(bins=30, ax=ax)
        st.pyplot(fig)

# Path: sl_app_pages/datacleanliness.py
# end of file
