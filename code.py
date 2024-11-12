from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn


class IrisDataProcessor:
  def __init__(self):
      iris = load_iris()# Load the iris dataset
      self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
      self.data['target'] = iris.target

  def prepare_data(self):
      scaler = StandardScaler() # Scale features
      X = scaler.fit_transform(self.data.drop(columns=['target']))
      y = self.data['target']
        
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
      return X_train, X_test, y_train, y_test

  def get_feature_stats(self):
      return self.data.describe() # Return basic statistical analysis


class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier()
        }

    def run_experiment(self):
        # Prepare data
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data()
        
        # Start a main MLflow run
        with mlflow.start_run():
            for model_name, model in self.models.items():
                # Create a nested run for each model to avoid parameter name conflicts
                with mlflow.start_run(nested=True):
                    # Cross-validation scores
                    scores = cross_val_score(model, X_train, y_train, cv=5)
                    
                    # Train and predict
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(y_test, predictions, average="weighted")
                    recall = recall_score(y_test, predictions, average="weighted")
                    
                    # Log model name and metrics
                    mlflow.log_param("model", model_name)
                    mlflow.log_metric("cv_accuracy", scores.mean())
                    mlflow.log_metric("test_accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)

                    # Log the model
                    mlflow.sklearn.log_model(model, model_name)

class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment

    def quantize_model(self):
        # Perform quantization for Logistic Regression
        log_model = LogisticRegression(max_iter=100, solver="liblinear")
        X_train, X_test, y_train, y_test = self.experiment.data_processor.prepare_data()
        log_model.fit(X_train, y_train)
        log_predictions = log_model.predict(X_test)
        
        # Log quantized Logistic Regression accuracy
        mlflow.log_metric("quantized_logistic_accuracy", accuracy_score(y_test, log_predictions))
        
        # Perform quantization for RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        
        # Log quantized Random Forest accuracy
        mlflow.log_metric("quantized_random_forest_accuracy", accuracy_score(y_test, rf_predictions))

    def run_tests(self):
        # Simple unit tests for both models
        X_train, X_test, y_train, y_test = self.experiment.data_processor.prepare_data()
        
        # Test for quantized Logistic Regression
        log_model = LogisticRegression(max_iter=100, solver="liblinear")
        log_model.fit(X_train, y_train)
        log_predictions = log_model.predict(X_test)
        assert accuracy_score(y_test, log_predictions) > 0.5, "Quantized Logistic Regression model accuracy is too low"
        
        # Test for quantized Random Forest
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        assert accuracy_score(y_test, rf_predictions) > 0.5, "Quantized Random Forest model accuracy is too low"


def main():
    # Initialize processor
    processor = IrisDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()
    
    # Run experiments
    experiment = IrisExperiment(processor)
    experiment.run_experiment()
    
    # Optimize and test
    optimizer = IrisModelOptimizer(experiment)
    optimizer.quantize_model()
    optimizer.run_tests()

if __name__ == "__main__":
    main()        