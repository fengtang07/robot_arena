import os
import mlflow
from stable_baselines3 import PPO
from environment import WarehouseEnv

# Set proxy bypass for MLflow connection
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Default Robot Training")

print("Training the default robot... This will take a few minutes.")

# --- Setup ---
env = WarehouseEnv()
env.reset()
os.makedirs("models", exist_ok=True)

# ENHANCED: Improved hyperparameters for better navigation learning
default_params = {
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'total_timesteps': 500000  # Increased from 200000 to 500000 for better learning
}

# --- Start MLflow Run ---
with mlflow.start_run(run_name="DefaultBaseline"):
    # Log parameters
    mlflow.log_params(default_params)
    
    # --- PPO Model ---
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=default_params['learning_rate'],
        n_steps=default_params['n_steps'],
        batch_size=default_params['batch_size'],
        n_epochs=default_params['n_epochs'],
        gamma=default_params['gamma'],
        verbose=1 # Set to 1 to see training progress
    )

    # --- Train and Save ---
    model.learn(total_timesteps=default_params['total_timesteps'])
    model_path = "models/default_robot_model.zip"
    model.save(model_path)

    # Log the trained model as an artifact in MLflow
    mlflow.log_artifact(model_path, "model")
    
    mlflow.log_metric("timesteps_trained", default_params['total_timesteps'])
    print("\nDefault robot trained, saved, and logged to MLflow.")