"""
Training scripts for Policy Gradient methods:
- REINFORCE
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)

Using Stable-Baselines3 with extensive hyperparameter tuning.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
# Note: REINFORCE is not directly available in SB3, so we approximate it using PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import sys

# Add parent directory to path to import custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import HydroponicsEnv


def train_ppo(hyperparams: dict, log_dir: str, model_save_path: str,
              total_timesteps: int = 100000):
    """Train a PPO model with specified hyperparameters."""
    env = HydroponicsEnv(grid_size=8, max_steps=200)
    env = Monitor(env, log_dir)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get('learning_rate', 3e-4),
        n_steps=hyperparams.get('n_steps', 2048),
        batch_size=hyperparams.get('batch_size', 64),
        n_epochs=hyperparams.get('n_epochs', 10),
        gamma=hyperparams.get('gamma', 0.99),
        gae_lambda=hyperparams.get('gae_lambda', 0.95),
        clip_range=hyperparams.get('clip_range', 0.2),
        ent_coef=hyperparams.get('ent_coef', 0.01),
        vf_coef=hyperparams.get('vf_coef', 0.5),
        max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
        policy_kwargs=dict(
            net_arch=hyperparams.get('net_arch', [256, 256])
        ),
        verbose=1,
        tensorboard_log=log_dir
    )
    
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
        name_prefix='ppo_model'
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save(os.path.join(model_save_path, 'final_model'))
    df = load_results(log_dir)
    return model, df


def train_a2c(hyperparams: dict, log_dir: str, model_save_path: str,
              total_timesteps: int = 100000):
    """Train an A2C model with specified hyperparameters."""
    env = HydroponicsEnv(grid_size=8, max_steps=200)
    env = Monitor(env, log_dir)
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get('learning_rate', 7e-4),
        n_steps=hyperparams.get('n_steps', 5),
        gamma=hyperparams.get('gamma', 0.99),
        gae_lambda=hyperparams.get('gae_lambda', 1.0),
        ent_coef=hyperparams.get('ent_coef', 0.01),
        vf_coef=hyperparams.get('vf_coef', 0.5),
        max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
        policy_kwargs=dict(
            net_arch=hyperparams.get('net_arch', [256, 256])
        ),
        verbose=1,
        tensorboard_log=log_dir
    )
    
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
        name_prefix='a2c_model'
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save(os.path.join(model_save_path, 'final_model'))
    df = load_results(log_dir)
    return model, df


def train_reinforce(hyperparams: dict, log_dir: str, model_save_path: str,
                    total_timesteps: int = 100000):
    """
    Train a REINFORCE model with specified hyperparameters.
    
    Note: Stable-Baselines3 doesn't have a native REINFORCE implementation.
    We approximate REINFORCE using PPO with REINFORCE-like settings:
    - Single epoch per update (no multiple passes)
    - No clipping (clip_range=1.0)
    - No value function (vf_coef=0.0)
    - No GAE (gae_lambda=1.0, which makes it equivalent to REINFORCE)
    - Full episode returns (n_steps matches episode length)
    """
    env = HydroponicsEnv(grid_size=8, max_steps=200)
    env = Monitor(env, log_dir)
    
    # Approximate REINFORCE using PPO with specific settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get('learning_rate', 1e-3),
        n_steps=hyperparams.get('n_steps', 200),  # Shorter episodes
        batch_size=hyperparams.get('batch_size', 200),
        n_epochs=hyperparams.get('n_epochs', 1),  # Single epoch like REINFORCE
        gamma=hyperparams.get('gamma', 0.99),
        gae_lambda=hyperparams.get('gae_lambda', 1.0),  # No GAE
        clip_range=hyperparams.get('clip_range', 1.0),  # No clipping
        ent_coef=hyperparams.get('ent_coef', 0.01),
        vf_coef=hyperparams.get('vf_coef', 0.0),  # No value function
        max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
        policy_kwargs=dict(
            net_arch=hyperparams.get('net_arch', [256, 256])
        ),
        verbose=1,
        tensorboard_log=log_dir
    )
    
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
        name_prefix='reinforce_model'
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save(os.path.join(model_save_path, 'final_model'))
    df = load_results(log_dir)
    return model, df


def hyperparameter_tuning_ppo():
    """Extensive hyperparameter tuning for PPO."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'ppo_tuning')
    os.makedirs(results_dir, exist_ok=True)
    
    configs = [
        # Config 1: Default
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 2: Higher LR
        {'learning_rate': 1e-3, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 3: Lower LR
        {'learning_rate': 1e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 4: More epochs
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 20,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 5: Larger batch
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 6: Tighter clipping
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.1, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 7: Higher entropy
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.05,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 8: Deeper network
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [512, 512, 256]},
        # Config 9: Longer steps
        {'learning_rate': 3e-4, 'n_steps': 4096, 'batch_size': 64, 'n_epochs': 10,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 10: Optimized
        {'learning_rate': 5e-4, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 15,
         'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.02,
         'vf_coef': 0.5, 'net_arch': [512, 256]},
        # Config 11: Conservative
        {'learning_rate': 1e-4, 'n_steps': 4096, 'batch_size': 128, 'n_epochs': 20,
         'gamma': 0.995, 'gae_lambda': 0.98, 'clip_range': 0.15, 'ent_coef': 0.01,
         'vf_coef': 0.5, 'net_arch': [256, 256]},
        # Config 12: Aggressive
        {'learning_rate': 1e-3, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 5,
         'gamma': 0.98, 'gae_lambda': 0.9, 'clip_range': 0.3, 'ent_coef': 0.05,
         'vf_coef': 0.3, 'net_arch': [128, 128]}
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Training PPO Config {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        config_dir = os.path.join(results_dir, f'config_{i+1}')
        log_dir = os.path.join(config_dir, 'logs')
        model_dir = os.path.join(config_dir, 'models')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            model, df = train_ppo(config, log_dir, model_dir, total_timesteps=50000)
            x, y = ts2xy(df, 'timesteps')
            if len(y) > 0:
                final_reward = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)
                results.append({
                    'config': i+1,
                    'hyperparams': config,
                    'final_reward': final_reward,
                    'max_reward': np.max(y),
                    'mean_reward': np.mean(y),
                    'model_path': os.path.join(model_dir, 'best', 'best_model')
                })
                print(f"Config {i+1} - Final Reward: {final_reward:.2f}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if results:
        best = max(results, key=lambda x: x['final_reward'])
        print(f"\nBest PPO Config: {best['config']}, Reward: {best['final_reward']:.2f}")
        import json
        with open(os.path.join(results_dir, 'tuning_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return best
    return None


def hyperparameter_tuning_a2c():
    """Extensive hyperparameter tuning for A2C."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'a2c_tuning')
    os.makedirs(results_dir, exist_ok=True)
    
    configs = [
        # Config 1-12 similar structure to PPO but with A2C-specific params
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 5e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 7e-4, 'n_steps': 10, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.995, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 0.95,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.05, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [512, 512, 256]},
        {'learning_rate': 7e-4, 'n_steps': 5, 'gamma': 0.99, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.3, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
         'ent_coef': 0.02, 'vf_coef': 0.5, 'net_arch': [512, 256]},
        {'learning_rate': 5e-4, 'n_steps': 10, 'gamma': 0.995, 'gae_lambda': 1.0,
         'ent_coef': 0.01, 'vf_coef': 0.5, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 3, 'gamma': 0.98, 'gae_lambda': 0.9,
         'ent_coef': 0.05, 'vf_coef': 0.3, 'net_arch': [128, 128]}
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Training A2C Config {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        config_dir = os.path.join(results_dir, f'config_{i+1}')
        log_dir = os.path.join(config_dir, 'logs')
        model_dir = os.path.join(config_dir, 'models')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            model, df = train_a2c(config, log_dir, model_dir, total_timesteps=50000)
            x, y = ts2xy(df, 'timesteps')
            if len(y) > 0:
                final_reward = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)
                results.append({
                    'config': i+1,
                    'hyperparams': config,
                    'final_reward': final_reward,
                    'max_reward': np.max(y),
                    'mean_reward': np.mean(y),
                    'model_path': os.path.join(model_dir, 'best', 'best_model')
                })
                print(f"Config {i+1} - Final Reward: {final_reward:.2f}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if results:
        best = max(results, key=lambda x: x['final_reward'])
        print(f"\nBest A2C Config: {best['config']}, Reward: {best['final_reward']:.2f}")
        import json
        with open(os.path.join(results_dir, 'tuning_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return best
    return None


def hyperparameter_tuning_reinforce():
    """Extensive hyperparameter tuning for REINFORCE."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'reinforce_tuning')
    os.makedirs(results_dir, exist_ok=True)
    
    configs = [
        {'learning_rate': 1e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 5e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 5e-4, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 100, 'batch_size': 100, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.995, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.05,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 1e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [512, 512, 256]},
        {'learning_rate': 1e-3, 'n_steps': 400, 'batch_size': 400, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 2e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.02,
         'vf_coef': 0.0, 'net_arch': [512, 256]},
        {'learning_rate': 1e-3, 'n_steps': 200, 'batch_size': 200, 'n_epochs': 1,
         'gamma': 0.99, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [128, 128]},
        {'learning_rate': 5e-4, 'n_steps': 400, 'batch_size': 400, 'n_epochs': 1,
         'gamma': 0.995, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.01,
         'vf_coef': 0.0, 'net_arch': [256, 256]},
        {'learning_rate': 5e-3, 'n_steps': 100, 'batch_size': 100, 'n_epochs': 1,
         'gamma': 0.98, 'gae_lambda': 1.0, 'clip_range': 1.0, 'ent_coef': 0.05,
         'vf_coef': 0.0, 'net_arch': [128, 128]}
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Training REINFORCE Config {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        config_dir = os.path.join(results_dir, f'config_{i+1}')
        log_dir = os.path.join(config_dir, 'logs')
        model_dir = os.path.join(config_dir, 'models')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            model, df = train_reinforce(config, log_dir, model_dir, total_timesteps=50000)
            x, y = ts2xy(df, 'timesteps')
            if len(y) > 0:
                final_reward = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)
                results.append({
                    'config': i+1,
                    'hyperparams': config,
                    'final_reward': final_reward,
                    'max_reward': np.max(y),
                    'mean_reward': np.mean(y),
                    'model_path': os.path.join(model_dir, 'best', 'best_model')
                })
                print(f"Config {i+1} - Final Reward: {final_reward:.2f}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if results:
        best = max(results, key=lambda x: x['final_reward'])
        print(f"\nBest REINFORCE Config: {best['config']}, Reward: {best['final_reward']:.2f}")
        import json
        with open(os.path.join(results_dir, 'tuning_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return best
    return None


if __name__ == "__main__":
    print("Starting Policy Gradient Hyperparameter Tuning...")
    
    print("\n" + "="*60)
    print("Training PPO...")
    print("="*60)
    best_ppo = hyperparameter_tuning_ppo()
    
    print("\n" + "="*60)
    print("Training A2C...")
    print("="*60)
    best_a2c = hyperparameter_tuning_a2c()
    
    print("\n" + "="*60)
    print("Training REINFORCE...")
    print("="*60)
    best_reinforce = hyperparameter_tuning_reinforce()
    
    # Train final models with best configs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if best_ppo:
        print(f"\nTraining final PPO model...")
        final_log = os.path.join(base_dir, 'results', 'ppo_final', 'logs')
        final_model_dir = os.path.join(base_dir, 'models', 'pg', 'ppo')
        os.makedirs(final_log, exist_ok=True)
        os.makedirs(final_model_dir, exist_ok=True)
        train_ppo(best_ppo['hyperparams'], final_log, final_model_dir, total_timesteps=200000)
    
    if best_a2c:
        print(f"\nTraining final A2C model...")
        final_log = os.path.join(base_dir, 'results', 'a2c_final', 'logs')
        final_model_dir = os.path.join(base_dir, 'models', 'pg', 'a2c')
        os.makedirs(final_log, exist_ok=True)
        os.makedirs(final_model_dir, exist_ok=True)
        train_a2c(best_a2c['hyperparams'], final_log, final_model_dir, total_timesteps=200000)
    
    if best_reinforce:
        print(f"\nTraining final REINFORCE model...")
        final_log = os.path.join(base_dir, 'results', 'reinforce_final', 'logs')
        final_model_dir = os.path.join(base_dir, 'models', 'pg', 'reinforce')
        os.makedirs(final_log, exist_ok=True)
        os.makedirs(final_model_dir, exist_ok=True)
        train_reinforce(best_reinforce['hyperparams'], final_log, final_model_dir, total_timesteps=200000)
    
    print("\nAll Policy Gradient training completed!")

