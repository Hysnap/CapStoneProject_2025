import streamlit as st
import pandas as pd
from sl_visualisations.map_visualisation_v2 import display_maps
from sl_visualisations.map_visualisation import display_map
# from d_transform.ML_model2 import evaluate_model, train_model
# from c_data_extract_combine.ETL import data_pipeline


def main():
    st.title("ML Model Analysis and Visualization")

    # Load data.articlesformap.csv
    st.header("Data Loading")
    data = pd.read_csv("data//articlesformap.csv")

    if data is None or data.empty:
        st.error("Failed to load data. Check ETL process.")
        return

    # Display Map
    display_maps()

    # # Train and Evaluate Model
    # st.header("Model Performance")
    # model_type = st.radio("Select Model Type:",
    # ("classification", "regression"))
    # target_column = "realness_score" if
    # model_type == "regression" else "label"

    # X = data.drop(columns=[target_column])
    # y = data[target_column]
    # model = train_model(X, y, model_type=model_type)

    # st.write("### Model Evaluation")
    # results = evaluate_model(model, X, y, model_type=model_type)
    # st.json(results)


if __name__ == "__main__":
    main()
