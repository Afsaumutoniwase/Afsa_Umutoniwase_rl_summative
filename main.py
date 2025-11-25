"""
Main entry point for running the best performing RL model.
Supports running trained models and generating visualizations.
"""

import os
import sys
import argparse
import numpy as np
import pygame
from stable_baselines3 import DQN, PPO, A2C

# Add environment to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import HydroponicsEnv
from environment.rendering import HydroponicsRenderer, create_env_wrapper


def run_random_agent(num_steps: int = 50, save_frames: bool = True):
    """
    Run the environment with a random agent (no training).
    This demonstrates the visualization without any model.
    
    Args:
        num_steps: Number of steps to run
        save_frames: Whether to save frames as images
    """
    print("Running Random Agent (No Training)")
    
    env = HydroponicsEnv(grid_size=8, max_steps=200)
    renderer = HydroponicsRenderer(env.grid_size)
    
    obs, info = env.reset()
    total_reward = 0
    
    frames_dir = "static_random_agent"
    os.makedirs(frames_dir, exist_ok=True)
    
    for step in range(num_steps):
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Prepare state for rendering
        env_state = {
            'plant_grid': env.plant_grid,
            'ec': env.ec,
            'ph': env.pH,
            'water_level': env.water_level,
            'light_intensity': env.light_intensity,
            'temperature': env.temperature,
            'humidity': env.humidity,
            'mature_plants': int(np.sum(env.plant_grid >= 0.95)),
            'avg_growth': float(np.mean(env.plant_grid)),
            'total_harvested': env.total_harvested
        }
        
        # Render
        renderer.render(env_state, action, reward, step)
        
        # Save frame if requested
        if save_frames and step % 5 == 0:  # Save every 5th frame
            frame_path = os.path.join(frames_dir, f"frame_{step:04d}.png")
            renderer.save_frame(frame_path, env_state, action, reward, step)
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return
        
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, "
              f"Total Reward={total_reward:.2f}, Mature={env_state['mature_plants']}, "
              f"Harvested={env.total_harvested}, AvgGrowth={env_state['avg_growth']:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            obs, info = env.reset()
            total_reward = 0
    
    print(f"\nRandom agent completed {num_steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    
    # Save final frame
    if save_frames:
        final_frame = os.path.join(frames_dir, "final_frame.png")
        renderer.save_frame(final_frame, env_state, None, None, num_steps)
        print(f"Frames saved to {frames_dir}/")
    
    # Keep window open for a moment
    pygame.time.wait(2000)
    renderer.close()


def load_best_model(algorithm: str, model_path: str = None):
    """
    Load the best trained model for a given algorithm.
    
    Args:
        algorithm: Algorithm name ('dqn', 'ppo', 'a2c', 'reinforce')
        model_path: Optional custom path to model
        
    Returns:
        Loaded model
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if algorithm.lower() == 'dqn':
            model_path = os.path.join(base_dir, 'models', 'dqn', 'best', 'best_model')
        elif algorithm.lower() == 'ppo':
            model_path = os.path.join(base_dir, 'models', 'pg', 'ppo', 'best', 'best_model')
        elif algorithm.lower() == 'a2c':
            model_path = os.path.join(base_dir, 'models', 'pg', 'a2c', 'best', 'best_model')
        elif algorithm.lower() == 'reinforce':
            model_path = os.path.join(base_dir, 'models', 'pg', 'reinforce', 'best', 'best_model')
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Try to load, fallback to final_model if best_model doesn't exist
    if not os.path.exists(model_path + '.zip'):
        model_path = model_path.replace('best', 'final_model')
        if not os.path.exists(model_path + '.zip'):
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading {algorithm.upper()} model from {model_path}...")
    
    if algorithm.lower() == 'dqn':
        model = DQN.load(model_path)
    elif algorithm.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algorithm.lower() == 'a2c':
        model = A2C.load(model_path)
    elif algorithm.lower() == 'reinforce':
        # REINFORCE is implemented as PPO variant
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def run_trained_agent(algorithm: str = 'dqn', num_episodes: int = 5, 
                     model_path: str = None, verbose: bool = True):
    """
    Run the environment with a trained agent.
    
    Args:
        algorithm: Algorithm to use ('dqn', 'ppo', 'a2c', 'reinforce')
        num_episodes: Number of episodes to run
        model_path: Optional custom path to model
        verbose: Whether to print detailed information
    """
    print(f"Running Trained {algorithm.upper()} Agent")
    
    # Load model
    try:
        model = load_best_model(algorithm, model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first using training scripts.")
        return
    
    # Create environment with rendering
    env = HydroponicsEnv(grid_size=8, max_steps=200, render_mode='human')
    renderer = HydroponicsRenderer(env.grid_size)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        if verbose:
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Prepare state for rendering
            env_state = {
                'plant_grid': env.plant_grid,
                'ec': env.ec,
                'ph': env.pH,
                'water_level': env.water_level,
                'light_intensity': env.light_intensity,
                'temperature': env.temperature,
                'humidity': env.humidity,
                'mature_plants': int(np.sum(env.plant_grid >= 0.90)),
                'avg_growth': float(np.mean(env.plant_grid)),
                'total_harvested': env.total_harvested
            }
            
            # Render
            renderer.render(env_state, action, reward, episode_length)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.close()
                    return
            
            if verbose and episode_length % 20 == 0:
                print(f"  Step {episode_length}: Action={action}, "
                      f"Reward={reward:.2f}, EC={env.ec:.2f}, pH={env.pH:.2f}, "
                      f"Mature={env_state['mature_plants']}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose:
            print(f"Episode {episode + 1} completed:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Total Harvested: {env.total_harvested}")
            print(f"  Avg Growth: {np.mean(env.plant_grid):.2%}")
    
    # Print summary statistics
    print("Performance Summary")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
    
    # Keep window open
    print("\nPress any key to close...")
    pygame.time.wait(5000)
    renderer.close()


def compare_all_models(num_episodes: int = 3):
    """Compare all trained models."""
    algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Evaluating {algo.upper()}")
        print(f"{'='*60}")
        
        try:
            model = load_best_model(algo)
            env = HydroponicsEnv(grid_size=8, max_steps=200)
            
            episode_rewards = []
            for episode in range(num_episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            results[algo] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards)
            }
            
            print(f"Mean Reward: {results[algo]['mean_reward']:.2f} ± {results[algo]['std_reward']:.2f}")
        except FileNotFoundError:
            print(f"Model not found for {algo}, skipping...")
            continue
    
    # Print comparison
    print("Model Comparison Summary")
    for algo, stats in results.items():
        print(f"{algo.upper():12} - Mean: {stats['mean_reward']:7.2f} ± {stats['std_reward']:5.2f} "
              f"(Max: {stats['max_reward']:7.2f}, Min: {stats['min_reward']:7.2f})")
    
    if results:
        best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])[0]
        print(f"\nBest Performing Algorithm: {best_algo.upper()}")
        return best_algo
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FarmSmart RL Agent')
    parser.add_argument('--mode', type=str, default='trained',
                       choices=['random', 'trained', 'compare'],
                       help='Mode: random (no model), trained (use model), or compare (all models)')
    parser.add_argument('--algorithm', type=str, default='dqn',
                       choices=['dqn', 'ppo', 'a2c', 'reinforce'],
                       help='Algorithm to use (for trained mode)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Custom path to model file')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        run_random_agent(num_steps=50, save_frames=True)
    elif args.mode == 'trained':
        run_trained_agent(
            algorithm=args.algorithm,
            num_episodes=args.episodes,
            model_path=args.model_path,
            verbose=not args.no_verbose
        )
    elif args.mode == 'compare':
        best = compare_all_models(num_episodes=args.episodes)
        if best:
            print(f"\nRunning best model ({best}) with visualization...")
            run_trained_agent(algorithm=best, num_episodes=3, verbose=True)




