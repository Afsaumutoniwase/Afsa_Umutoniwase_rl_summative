"""
Training script for Deep Q-Network (DQN) using Stable-Baselines3.
Includes hyperparameter tuning and model evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import sys

# Add parent directory to path to import custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import HydroponicsEnv


def train_dqn(hyperparams: dict, log_dir: str, model_save_path: str, 
              total_timesteps: int = 100000):
    """
    Train a DQN model with specified hyperparameters.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        log_dir: Directory for logging
        model_save_path: Path to save the trained model
        total_timesteps: Total training timesteps
        
    Returns:
        Trained model and training statistics
    """
    # Create environment
    env = HydroponicsEnv(grid_size=8, max_steps=200)
    env = Monitor(env, log_dir)
    
    # Create model with hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get('learning_rate', 1e-4),
        buffer_size=hyperparams.get('buffer_size', 100000),
        learning_starts=hyperparams.get('learning_starts', 1000),
        batch_size=hyperparams.get('batch_size', 32),
        tau=hyperparams.get('tau', 1.0),
        gamma=hyperparams.get('gamma', 0.99),
        train_freq=hyperparams.get('train_freq', 4),
        gradient_steps=hyperparams.get('gradient_steps', 1),
        target_update_interval=hyperparams.get('target_update_interval', 1000),
        exploration_fraction=hyperparams.get('exploration_fraction', 0.1),
        exploration_initial_eps=hyperparams.get('exploration_initial_eps', 1.0),
        exploration_final_eps=hyperparams.get('exploration_final_eps', 0.05),
        policy_kwargs=dict(
            net_arch=hyperparams.get('net_arch', [256, 256])
        ),
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Setup callbacks
    eval_env = HydroponicsEnv(grid_size=8, max_steps=200)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_path, 'best'),
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(model_save_path, 'checkpoints'),
        name_prefix='dqn_model'
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(os.path.join(model_save_path, 'final_model'))
    
    # Load and return training statistics
    df = load_results(log_dir)
    return model, df


def hyperparameter_tuning():
    """
    Perform extensive hyperparameter tuning for DQN.
    Tests at least 10 different hyperparameter combinations.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'dqn_tuning')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define hyperparameter search space
    hyperparameter_configs = [
        # Config 1: Default baseline
        {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 2: Higher learning rate
        {
            'learning_rate': 5e-4,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 3: Lower learning rate
        {
            'learning_rate': 5e-5,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 4: Larger buffer
        {
            'learning_rate': 1e-4,
            'buffer_size': 200000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 5: Larger batch size
        {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 6: Higher discount factor
        {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.995,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 7: More exploration
        {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.2,
            'exploration_final_eps': 0.1,
            'target_update_interval': 1000,
            'net_arch': [256, 256]
        },
        # Config 8: Deeper network
        {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'net_arch': [512, 512, 256]
        },
        # Config 9: Faster target update
        {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'target_update_interval': 500,
            'net_arch': [256, 256]
        },
        # Config 10: Combined optimal (tuned)
        {
            'learning_rate': 3e-4,
            'buffer_size': 150000,
            'batch_size': 64,
            'gamma': 0.99,
            'exploration_fraction': 0.15,
            'exploration_final_eps': 0.08,
            'target_update_interval': 750,
            'net_arch': [512, 256]
        },
        # Config 11: Conservative learning
        {
            'learning_rate': 5e-5,
            'buffer_size': 200000,
            'batch_size': 64,
            'gamma': 0.995,
            'exploration_fraction': 0.2,
            'exploration_final_eps': 0.1,
            'target_update_interval': 1500,
            'net_arch': [256, 256]
        },
        # Config 12: Aggressive learning
        {
            'learning_rate': 1e-3,
            'buffer_size': 50000,
            'batch_size': 16,
            'gamma': 0.98,
            'exploration_fraction': 0.05,
            'exploration_final_eps': 0.02,
            'target_update_interval': 500,
            'net_arch': [128, 128]
        }
    ]
    
    results = []
    
    for i, config in enumerate(hyperparameter_configs):
        print(f"\n{'='*60}")
        print(f"Training DQN with Configuration {i+1}/{len(hyperparameter_configs)}")
        print(f"{'='*60}")
        print(f"Hyperparameters: {config}")
        
        config_dir = os.path.join(results_dir, f'config_{i+1}')
        log_dir = os.path.join(config_dir, 'logs')
        model_dir = os.path.join(config_dir, 'models')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            model, df = train_dqn(
                config,
                log_dir,
                model_dir,
                total_timesteps=50000  # Reduced for faster tuning
            )
            
            # Calculate metrics
            x, y = ts2xy(df, 'timesteps')
            if len(y) > 0:
                final_reward = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)
                max_reward = np.max(y)
                mean_reward = np.mean(y)
                
                results.append({
                    'config': i+1,
                    'hyperparams': config,
                    'final_reward': final_reward,
                    'max_reward': max_reward,
                    'mean_reward': mean_reward,
                    'model_path': os.path.join(model_dir, 'best', 'best_model')
                })
                
                print(f"Config {i+1} Results:")
                print(f"  Final Reward (avg last 100): {final_reward:.2f}")
                print(f"  Max Reward: {max_reward:.2f}")
                print(f"  Mean Reward: {mean_reward:.2f}")
        except Exception as e:
            print(f"Error training config {i+1}: {e}")
            continue
    
    # Find best configuration
    if results:
        best_config = max(results, key=lambda x: x['final_reward'])
        print(f"\n{'='*60}")
        print(f"Best Configuration: {best_config['config']}")
        print(f"Final Reward: {best_config['final_reward']:.2f}")
        print(f"Hyperparameters: {best_config['hyperparams']}")
        print(f"{'='*60}")
        
        # Save results summary
        import json
        with open(os.path.join(results_dir, 'tuning_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return best_config
    else:
        print("No successful training runs!")
        return None


if __name__ == "__main__":
    print("Starting DQN Hyperparameter Tuning...")
    best_config = hyperparameter_tuning()
    
    if best_config:
        print(f"\nTraining final model with best configuration...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        final_log_dir = os.path.join(base_dir, 'results', 'dqn_final', 'logs')
        final_model_dir = os.path.join(base_dir, 'models', 'dqn')
        os.makedirs(final_log_dir, exist_ok=True)
        os.makedirs(final_model_dir, exist_ok=True)
        
        final_model, _ = train_dqn(
            best_config['hyperparams'],
            final_log_dir,
            final_model_dir,
            total_timesteps=200000  # Longer training for final model
        )
        print("Final DQN model training completed!")

