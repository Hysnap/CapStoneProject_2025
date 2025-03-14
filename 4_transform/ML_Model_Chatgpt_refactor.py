import numpy as np
import pandas as pd
import seaborn as sns
import csv
from scipy.sparse import issparse
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sns.set_style("whitegrid")


def preprocess_data(source_df, low_cardinality=None, skewed_numeric=None,
                    numeric=None, high_cardinality=None,
                    target=None, run_name=None):
    """Preprocesses the data by encoding categorical features
        and scaling numeric features."""
    transformers = []

    if skewed_numeric:
        transformers.append(('num_skew',
                             PowerTransformer(method='yeo-johnson',
                                              standardize=True),
                             skewed_numeric))
    if numeric:
        transformers.append(('num',
                             StandardScaler(),
                             numeric))
    if high_cardinality:
        transformers.append(('ord',
                             OrdinalEncoder(handle_unknown="use_encoded_value",
                                            unknown_value=-1),
                             high_cardinality))
    if low_cardinality:
        transformers.append(('cat',
                             OneHotEncoder(handle_unknown='ignore'),
                             low_cardinality))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([("preprocessor", preprocessor)])

    transformed_data = pipeline.fit_transform(source_df)

    if issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    feature_names = []
    if low_cardinality:
        feature_names.extend(
            pipeline.named_steps['preprocessor']
            .named_transformers_['cat']
            .get_feature_names_out(low_cardinality)
        )
    if skewed_numeric:
        feature_names.extend(skewed_numeric)
    if numeric:
        feature_names.extend(numeric)
    if high_cardinality:
        feature_names.extend(high_cardinality)

    processed_df = pd.DataFrame(transformed_data, columns=feature_names)
    processed_df[target] = source_df[target].values
    return processed_df, pipeline


def check_correlation(correl_df, target):
    """Returns correlation of features with the target variable."""
    return correl_df.corr()[target].sort_values(ascending=False)


def train_model(xt_tm_df, yt_tm_df):
    """Trains a RandomForest model and returns feature importances."""
    X = xt_tm_df
    y = yt_tm_df
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = (
        pd.Series(model.feature_importances_,
                  index=X.columns).sort_values(ascending=False))
    return model, importances


def feature_selection(fs_df,
                      target,
                      var_threshold=0.01,
                      corr_threshold=0.9):
    """Removes low-variance and highly correlated features."""

    selector = VarianceThreshold(threshold=var_threshold)
    X_reduced = selector.fit_transform(fs_df)
    selected_features = fs_df.columns[selector.get_support()]

    corr_matrix = fs_df[selected_features].corr().abs()
    upper = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k=1).astype(bool)))
    to_drop = (
        [column for column in upper.columns
         if any(upper[column] > corr_threshold)])
    X_final = fs_df[selected_features].drop(columns=to_drop)

    return X_final, list(X_final.columns)


def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluates model performance using confusion
        matrix and classification report."""
    results = {}
    for name, X, y in zip(['train', 'test'],
                          [X_train, X_test],
                          [y_train, y_test]):
        y_pred = model.predict(X)
        results[f'{name}_conf_matrix'] = confusion_matrix(y, y_pred)
        results[f'{name}_report'] = (
            classification_report(y,
                                  y_pred,
                                  output_dict=True))
    return results


def save_results_to_csv(run_name,
                        numeric,
                        skewed_numeric,
                        high_cardinality,
                        low_cardinality,
                        target,
                        correlation,
                        selected_features,
                        importances,
                        evaluation_results,
                        csv_file="data/MachineLearning_results.csv"):
    # Prepare row data
    row_data = {
        "Run Name": run_name,
        "Run Date": str(pd.Timestamp.now()),
        "Target Variable": target,
        "Numeric Features": ", ".join(numeric),
        "Skewed Numeric": ", ".join(skewed_numeric),
        "High Cardinality Features": ", ".join(high_cardinality),
        "Low Cardinality Features": ", ".join(low_cardinality),
        "Selected Features": ", ".join(selected_features),
    }

    # Add correlation values (flatten dictionary)
    for feature, value in correlation.items():
        row_data[f"Correlation_{feature}"] = value

    # Add feature importances (flatten dictionary)
    for feature, importance in importances.items():
        row_data[f"Importance_{feature}"] = importance

    # Add evaluation metrics (flatten nested dictionary)
    for key, metrics in evaluation_results.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                row_data[f"Test_{key}_{metric_name}"] = value
        else:
            row_data[f"Test_{key}"] = metrics.tolist() if isinstance(metrics, np.ndarray) else metrics

    # Write to CSV
    file_exists = False
    try:
        with open(csv_file, "r") as f:
            file_exists = True
    except FileNotFoundError:
        pass  # File does not exist, it will be created

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()  # Write header only if file does not exist
        writer.writerow(row_data)


def main():
    """Main execution function."""
    source_df = pd.read_csv("data/combined_data.zip")
    print(source_df.head())
    print(source_df.info())
    run_name = "ML_Model_3"
    numeric = ["overall_subjectivity", "overall_polarity",
               "title_subjectivity", "title_polarity",
               "article_subjectivity", "article_polarity",
               "contradiction_polarity", "contradiction_subjectivity"]
    skewed_numeric = ["title_length", "text_length"]
    high_cardinality = ["sentiment_title",
                        "sentiment_article"]
    low_cardinality = ["media_type", "sentiment_overall"]
    target = "label"

    processed_df, pp_pipeline = preprocess_data(
        source_df, low_cardinality, skewed_numeric,
        numeric, high_cardinality, target, run_name
    )

    correlation = check_correlation(processed_df, target)

    (X_reduced,
     selected_features) = (
         feature_selection(processed_df.drop(columns=[target]),
                           target))

    (X_train,
     X_test,
     y_train,
     y_test) = split_data(X_reduced,
                          processed_df[target])

    # print("Processed DF Columns:", processed_df.columns)
    # print("First 5 rows:\n", processed_df.head())
    # print("X_reduced shape:", X_reduced.shape)
    # print("X_reduced columns:", X_reduced.columns)
    # print("X_reduced first 5 row:\n", X_reduced.head())

    model, importances = train_model(X_train, y_train)

    X_test = X_test[selected_features]

    evaluation_results = evaluate_model(model,
                                        X_train,
                                        y_train,
                                        X_test,
                                        y_test)

    save_results_to_csv(
        run_name=run_name,
        numeric=numeric,
        skewed_numeric=skewed_numeric,
        high_cardinality=high_cardinality,
        low_cardinality=low_cardinality,
        target=target,
        correlation=correlation.to_dict(),
        selected_features=selected_features,
        importances=importances.to_dict(),
        evaluation_results={key: (
            value.tolist() if isinstance(value, np.ndarray) else value)
                            for key, value in evaluation_results.items()}
        )


    # outputs = {"Machine_Learning_Models": {
    #     run_name: {
    #         "datetimerun": str(pd.Timestamp.now()),
    #         "numeric": numeric,
    #         "skewed_numeric": skewed_numeric,
    #         "high_cardinality": high_cardinality,
    #         "low_cardinality": low_cardinality,
    #         "target": target,
    #         "correlation": correlation.to_dict(),
    #         "selected_features": selected_features,
    #         "importances": importances.to_dict(),
    #         "evaluation": {
    #             key: (value.tolist() if isinstance(value,
    #                                                np.ndarray) else value)
    #             for key, value in evaluation_results.items()
    #         }
    #     }}}
    return evaluation_results


if __name__ == "__main__":
    # Simulated function call with example parameters (replace with real data)
    results = main()
    print(results)
