import numpy as np
import pandas as pd
import seaborn as sns
import json
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


def main():
    """Main execution function."""
    source_df = pd.read_csv("data/combined_data.zip")
    print(source_df.head())
    print(source_df.info())
    run_name = "ML_Model_1"
    numeric = ["month", "day", "year", "week_of_year",
               "is_weekend", "is_weekday"]
    skewed_numeric = ["title_length", "text_length"]
    high_cardinality = ["locationsfromarticle", "day_label"]
    low_cardinality = ["media_type", "source_name"]
    target = "label"

    processed_df, pp_pipeline = preprocess_data(
        source_df, low_cardinality, skewed_numeric,
        numeric, high_cardinality, target, run_name
    )

    correlation = check_correlation(processed_df, target)
    
    (X_reduced,
     selected_features) = feature_selection(processed_df.drop(columns=[target]),
                                            target)

    (X_train,
     X_test,
     y_train,
     y_test) = split_data(X_reduced,
                          processed_df[target])

    print("Processed DF Columns:", processed_df.columns)
    print("First 5 rows:\n", processed_df.head())
    print("X_reduced shape:", X_reduced.shape)
    print("X_reduced columns:", X_reduced.columns)
    print("X_reduced first 5 row:\n", X_reduced.head())

    model, importances = train_model(X_train, y_train)

    X_test = X_test[selected_features]

    evaluation_results = evaluate_model(model,
                                        X_train,
                                        y_train,
                                        X_test,
                                        y_test)

    outputs = {
        run_name: {
            "datetimerun": str(pd.Timestamp.now()),
            "numeric": numeric,
            "skewed_numeric": skewed_numeric,
            "high_cardinality": high_cardinality,
            "low_cardinality": low_cardinality,
            "target": target,
            "correlation": correlation.to_dict(),
            "selected_features": selected_features,
            "importances": importances.to_dict(),
            "evaluation": {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in evaluation_results.items()
            }
        }}

    return outputs


if __name__ == "__main__":
    results = main()
    # append results to a json file
    with open("data/results.json", "a") as f:
        json.dump(results, f, indent=4)
        f.write("\n")
