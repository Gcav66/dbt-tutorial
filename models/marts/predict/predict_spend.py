#predict_spend.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from joblib import dump
from datetime import datetime

def model(dbt, session):
    dbt.config(
        materialized="incremental",
        packages=["pandas", "scikit-learn", "joblib"]
    )

    cross_validaton_folds = 10
    number_of_folds = 5
    polynomial_features_degrees = 2
    train_accuracy_threshold = 0.85
    test_accuracy_threshold = 0.85
    save_model = True

    df = dbt.ref("dim_channel_spend").to_pandas()
    numeric_features = ['search_engine','social_media','video','email']
    numeric_transformer = Pipeline(steps=[('poly',PolynomialFeatures(degree = polynomial_features_degrees)),('scaler', StandardScaler())])

    # Combine the preprocessed step together using the Column Transformer module
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # The next step is the integrate the features we just preprocessed with our Machine Learning algorithm to enable us to build a model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LinearRegression())])
    parameteres = {}

    X = df.drop('REVENUE', axis = 1)
    y = df['REVENUE']

    # Split dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    # Use GridSearch to find the best fitting model based on number_of_folds folds
    model = GridSearchCV(pipeline, param_grid=parameteres, cv=number_of_folds)

    model.fit(X_train, y_train)
    train_r2_score = model.score(X_train, y_train)
    test_r2_score = model.score(X_test, y_test)
    if save_model:
        if train_r2_score >= train_accuracy_threshold and test_r2_score >= test_accuracy_threshold:
            # Upload trained model to a stage
            model_output_dir = '/tmp'
            model_file = os.path.join(model_output_dir, 'model.joblib')
            dump(model, model_file)
            session.sql("create stage if not exists gus_stg").collect()
            session.file.put(model_file, "@gus_stg",overwrite=True)
            model_saved = True

    # Return model R2 score on train and test data
    if dbt.is_incremental:
        result = {"datetime": str(datetime.now()),
            "R2 score on Train": train_r2_score,
            "R2 threshold on Train": train_accuracy_threshold,
            "R2 score on Test": test_r2_score,
            "R2 threshold on Test": test_accuracy_threshold,
            "Model saved": False}
        result_df = pd.DataFrame([result])
    else:
        result = {"datetime": str(datetime.now()),
            "R2 score on Train": train_r2_score,
            "R2 threshold on Train": train_accuracy_threshold,
            "R2 score on Test": test_r2_score,
            "R2 threshold on Test": test_accuracy_threshold,
            "Model saved": False}
        result_df = pd.DataFrame([result])

    return result_df