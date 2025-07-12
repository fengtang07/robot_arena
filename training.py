import os
import time
import mlflow
import mlflow.pytorch
from stable_baselines3 import PPO
from environment import WarehouseEnv
import warnings

def train_user_robot(hyperparams, username):
    """
    Train the PPO agent with user-defined hyperparameters and log to MLflow.
    Falls back to local storage if MLflow is unavailable.
    """
    mlflow_available = False
    
    # Try to connect to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        
        # Test connection with a simple API call
        import requests
        response = requests.get("http://127.0.0.1:5001/api/2.0/mlflow/experiments/list", timeout=10)
        if response.status_code == 200:
            mlflow_available = True
            print("✅ Connected to MLflow successfully")
        else:
            print(f"⚠️  MLflow server responded with status {response.status_code}, using local storage")
    except Exception as e:
        print(f"⚠️  MLflow connection failed: {e}")
        print("📁 Using local storage for model saving")
    
    # Start MLflow run if available
    if mlflow_available:
        try:
            mlflow.set_experiment("Warehouse Robot Competition")
            mlflow_run = mlflow.start_run(run_name=f"train_{username}")
            print(f"🔬 Started MLflow run: {mlflow_run.info.run_id}")
            
            # Log parameters
            mlflow.log_params(hyperparams)
            mlflow.set_tag("username", username)
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
            mlflow_available = False
    
    # Create environment and model
    print("🏭 Creating warehouse environment...")
    env = WarehouseEnv()
    env.reset()
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{username}_robot_model.zip"
    
    print("🤖 Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=hyperparams['learning_rate'],
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        n_epochs=hyperparams['n_epochs'],
        verbose=1  # Show training progress
    )
    
    print(f"🚀 Starting training for {hyperparams['total_timesteps']} timesteps...")
    start_time = time.time()
    
    # Train the model
    model.learn(total_timesteps=hyperparams['total_timesteps'])
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Save model locally
    print(f"💾 Saving model to: {model_path}")
    model.save(model_path)
    
    # Log to MLflow if available
    if mlflow_available:
        try:
            mlflow.log_metric("training_duration_sec", training_time)
            mlflow.log_artifact(model_path, "model")
            
            # Log the model with SB3 integration
            mlflow.stable_baselines3.log_model(
                sb3_model=model,
                artifact_path=f"user_model_{username}",
                registered_model_name=f"WarehouseRobot_{username}"
            )
            
            print("📊 Model logged to MLflow successfully")
            mlflow.end_run()
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
            print("📁 Model still saved locally")
    
    return {
        'time_taken': training_time,
        'model_path': model_path,
        'mlflow_logged': mlflow_available
    } 

def train_champion_robot():
    """Train the champion robot with improved environment settings"""
    print("🏆 Training Champion Robot with improved environment...")
    
    # Create environment with improved settings
    env = WarehouseEnv(max_steps=1000)
    
    # Enhanced hyperparameters for better performance
    enhanced_params = {
        'learning_rate': 0.0005,
        'n_steps': 4096,
        'batch_size': 128,
        'n_epochs': 10,
        'total_timesteps': 100000,
        'gamma': 0.99,
        'clip_range': 0.2
    }
    
    # Create and train the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=enhanced_params['learning_rate'],
        n_steps=enhanced_params['n_steps'],
        batch_size=enhanced_params['batch_size'],
        n_epochs=enhanced_params['n_epochs'],
        gamma=enhanced_params['gamma'],
        clip_range=enhanced_params['clip_range'],
        verbose=1
    )
    
    print(f"🚀 Starting training with {enhanced_params['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=enhanced_params['total_timesteps'])
    
    end_time = time.time()
    print(f"✅ Training completed in {end_time - start_time:.2f} seconds")
    
    # Save the improved model
    model_path = "models/champion_robot_model.zip"
    model.save(model_path)
    print(f"💾 Champion robot model saved to: {model_path}")
    
    return {
        'model_path': model_path,
        'training_time': end_time - start_time,
        'hyperparams': enhanced_params
    } 