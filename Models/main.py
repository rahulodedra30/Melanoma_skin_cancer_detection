import os
import mlflow
import mlflow.keras
from model import build_model, compile_model, load_data, train_model, evaluate_model

def get_train_test_paths():
    train_dir = os.path.join('', 'train')
    test_dir = os.path.join('', 'test')
    return train_dir, test_dir

def kill_existing_run():
    active_run = mlflow.active_run()
    if active_run is not None:
        run_id = active_run.info.run_id
        mlflow.end_run(run_id=run_id)

if __name__ == "__main__":
    # Set MLflow tracking URI
    # mlflow.set_tracking_uri('mlflow server --host 127.0.0.1 --port 5000')

    # Kill existing MLflow run if present
    kill_existing_run()

    # Start MLflow run
    with mlflow.start_run():
        # Load data
        train_dir, test_dir = get_train_test_paths()
        train_generator, test_generator = load_data(train_dir, test_dir)

        # Build model
        model = build_model()
        compile_model(model)

        # Train model
        history = train_model(model, train_generator, test_generator, epochs=10)

        # Evaluate model
        evaluate_model(model, test_generator)

        # Log artifacts
        mlflow.log_artifacts(train_dir, artifact_path="train_data")
        mlflow.log_artifacts(test_dir, artifact_path="test_data")



