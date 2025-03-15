import pandas as pd
from d_transform.ML_Model_Chatgpt_refactor import (
    preprocess_data,
    split_data)
from sklearn.ensemble import (RandomForestRegressor,
                              RandomForestClassifier)
from sklearn.metrics import (mean_squared_error,
                             classification_report,
                             confusion_matrix)


def adjust_realness_score(df, mode="classification"):
    if mode == "regression":
        # Fake -> 1, Real -> 5
        df['realness_score'] = df['label'].map({0: 1, 1: 5})
    return df


def train_model(X_train, y_train, model_type="classification"):
    if model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_type="classification"):
    y_pred = model.predict(X_test)

    if model_type == "classification":
        return {
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Classification Report': classification_report(y_test,
                                                           y_pred,
                                                           output_dict=True)
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        return {'Mean Squared Error': mse, 'Predictions': y_pred}


# Main workflow
# Change to "classification" for original binary model
model_type = "regression"
source_df = pd.read_csv("data/combined_data.zip")

# drop rows with missing values
source_df = source_df.dropna(inplace=True)

if model_type == "regression":
    source_df = adjust_realness_score(source_df, mode=model_type)
    target_column = "realness_score"
else:
    target_column = "label"

# Preprocessing
processed_df, _ = preprocess_data(
    source_df,
    low_cardinality=["media_type", "sentiment_overall"],
    skewed_numeric=["title_length", "text_length"],
    numeric=["overall_subjectivity", "overall_polarity"],
    high_cardinality=["sentiment_title", "sentiment_article"],
    target=target_column,
    run_name=f"ML_Model_{model_type.capitalize()}"
)

# Split data
X_train, X_test, y_train, y_test = split_data(
    processed_df.drop(columns=[target_column]),
    processed_df[target_column]
)

# Train model
model = train_model(X_train, y_train, model_type=model_type)

# Evaluate model
evaluation_results = evaluate_model(model,
                                    X_test,
                                    y_test,
                                    model_type=model_type)
print(evaluation_results)
