import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Atur URI tracking dan nama eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# Create a new MLflow Experiment
mlflow.set_experiment("Obesity Modeling - SVM")
 
X_train = pd.read_csv("data_preprocessing/X_train.csv")
X_test = pd.read_csv("data_preprocessing/X_test.csv")
y_train = pd.read_csv("data_preprocessing/y_train.csv")
y_test = pd.read_csv("data_preprocessing/y_test.csv")
 
input_example = X_train[0:5]
# Mulai MLflow run
with mlflow.start_run(run_name="SVM_Model"):
    # latih model 
    model = SVC()  
    model.fit(X_train, y_train)

    # Log model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Evaluasi dan log akurasi
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    print(f"Akurasi model (default SVC): {accuracy:.4f}")