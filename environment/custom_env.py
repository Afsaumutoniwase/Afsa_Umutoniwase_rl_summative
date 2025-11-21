"""
Custom Gymnasium Environment for FarmSmart Rwanda Hydroponics System
This environment simulates a hydroponic farming system where an AI agent
manages nutrient levels, pH, water, and other critical parameters.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
import math


class HydroponicsEnv(gym.Env):
    """
    A hydroponic farming environment where the agent manages a grid-based
    hydroponic system to optimize crop yield and resource efficiency.
    
    The agent must balance:
    - Nutrient concentration (EC - Electrical Conductivity)
    - pH levels
    - Water levels
    - Light intensity
    - Plant growth stages
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, grid_size=8, max_steps=200, render_mode=None):
        """
        Initialize the hydroponics environment.
        
        Args:
            grid_size: Size of the hydroponic grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        
        # Action space: 9 discrete actions
        # 0: Increase nutrients (EC)
        # 1: Decrease nutrients (EC)
        # 2: Increase pH
        # 3: Decrease pH
        # 4: Add water
        # 5: Increase light
        # 6: Decrease light
        # 7: Harvest mature plants
        # 8: Do nothing (wait)
        self.action_space = spaces.Discrete(9)
        
        # Observation space: Multi-dimensional continuous
        # [EC, pH, water_level, light_intensity, avg_growth_stage, 
        #  mature_plants_ratio, temperature, humidity, time_of_day]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 0.0]),
            high=np.array([4.0, 8.0, 100.0, 100.0, 1.0, 1.0, 35.0, 90.0, 1.0]),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize hydroponic system parameters
        # EC (Electrical Conductivity) - optimal range: 1.5-2.5
        self.ec = self.np_random.uniform(1.2, 2.0)
        
        # pH - optimal range: 5.5-6.5 for most crops
        self.pH = self.np_random.uniform(5.0, 7.0)
        
        # Water level (percentage)
        self.water_level = self.np_random.uniform(60.0, 80.0)
        
        # Light intensity (percentage)
        self.light_intensity = self.np_random.uniform(50.0, 70.0)
        
        # Plant grid: each cell has a growth stage (0.0 to 1.0)
        # Start with more mature plants so they can be harvested within episode
        self.plant_grid = self.np_random.uniform(0.5, 0.7, (self.grid_size, self.grid_size))
        
        # Environmental conditions
        self.temperature = self.np_random.uniform(20.0, 28.0)
        self.humidity = self.np_random.uniform(50.0, 70.0)
        self.time_of_day = 0.0  # 0.0 = morning, 1.0 = end of day
        
        # Track performance metrics
        self.total_harvested = 0
        self.optimal_conditions_count = 0
        self.last_harvest_count = 0  # Track plants harvested in last action
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-8)
            
        Returns:
            observation: Current state observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was cut short
            info: Additional information
        """
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Natural progression (plants grow, conditions change)
        self._natural_progression()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: int):
        """Apply the selected action to the environment."""
        self.last_harvest_count = 0  # Reset harvest count for this step
        
        if action == 0:  # Increase nutrients (EC)
            self.ec = np.clip(self.ec + 0.2, 0.0, 4.0)
        elif action == 1:  # Decrease nutrients (EC)
            self.ec = np.clip(self.ec - 0.2, 0.0, 4.0)
        elif action == 2:  # Increase pH
            self.pH = np.clip(self.pH + 0.1, 4.0, 8.0)
        elif action == 3:  # Decrease pH
            self.pH = np.clip(self.pH - 0.1, 4.0, 8.0)
        elif action == 4:  # Add water
            self.water_level = np.clip(self.water_level + 5.0, 0.0, 100.0)
        elif action == 5:  # Increase light
            self.light_intensity = np.clip(self.light_intensity + 10.0, 0.0, 100.0)
        elif action == 6:  # Decrease light
            self.light_intensity = np.clip(self.light_intensity - 10.0, 0.0, 100.0)
        elif action == 7:  # Harvest mature plants
            harvested = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.plant_grid[i, j] >= 0.90:  # Mature plant (lowered from 0.95 to 0.90)
                        self.plant_grid[i, j] = self.np_random.uniform(0.5, 0.7)  # Replant with head start
                        harvested += 1
            self.total_harvested += harvested
            self.last_harvest_count = harvested  # Track for reward calculation
        # action == 8: Do nothing (wait)
    
    def _natural_progression(self):
        """Simulate natural changes in the system."""
        # Plants grow based on conditions
        growth_rate = self._calculate_growth_rate()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Plants grow, but slower if conditions are poor
                # Increased to 0.025 for faster maturation (plants can mature in 10-20 steps)
                self.plant_grid[i, j] = np.clip(
                    self.plant_grid[i, j] + growth_rate * 0.025,
                    0.0, 1.0
                )
        
        # Water level decreases naturally
        self.water_level = np.clip(self.water_level - 0.5, 0.0, 100.0)
        
        # EC decreases slightly as plants consume nutrients
        if self.ec > 0.1:
            self.ec = np.clip(self.ec - 0.02, 0.0, 4.0)
        
        # Time progresses
        self.time_of_day = (self.time_of_day + 0.01) % 1.0
        
        # Temperature and humidity fluctuate slightly
        self.temperature += self.np_random.uniform(-0.5, 0.5)
        self.temperature = np.clip(self.temperature, 15.0, 35.0)
        
        self.humidity += self.np_random.uniform(-2.0, 2.0)
        self.humidity = np.clip(self.humidity, 30.0, 90.0)
    
    def _calculate_growth_rate(self) -> float:
        """
        Calculate plant growth rate based on environmental conditions.
        Optimal conditions yield higher growth rates.
        """
        # EC optimal range: 1.5-2.5
        ec_score = 1.0 - abs(self.ec - 2.0) / 1.0
        ec_score = np.clip(ec_score, 0.0, 1.0)
        
        # pH optimal range: 5.5-6.5
        ph_score = 1.0 - abs(self.pH - 6.0) / 0.5
        ph_score = np.clip(ph_score, 0.0, 1.0)
        
        # Water level optimal: 70-85%
        water_score = 1.0 - abs(self.water_level - 77.5) / 15.0
        water_score = np.clip(water_score, 0.0, 1.0)
        
        # Light optimal: 60-80%
        light_score = 1.0 - abs(self.light_intensity - 70.0) / 20.0
        light_score = np.clip(light_score, 0.0, 1.0)
        
        # Temperature optimal: 22-26°C
        temp_score = 1.0 - abs(self.temperature - 24.0) / 4.0
        temp_score = np.clip(temp_score, 0.0, 1.0)
        
        # Combined growth rate (weighted average)
        growth_rate = (ec_score * 0.25 + ph_score * 0.25 + water_score * 0.15 + 
                      light_score * 0.15 + temp_score * 0.20)
        
        return growth_rate
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on system state.
        Positive rewards for optimal conditions and harvests.
        Negative rewards for poor conditions.
        """
        reward = 0.0
        
        # Reward for optimal EC (1.5-2.5)
        if 1.5 <= self.ec <= 2.5:
            reward += 0.5
            self.optimal_conditions_count += 1
        elif self.ec < 0.5 or self.ec > 3.5:
            reward -= 1.0  # Severe nutrient imbalance
        
        # Reward for optimal pH (5.5-6.5)
        if 5.5 <= self.pH <= 6.5:
            reward += 0.5
        elif self.pH < 4.5 or self.pH > 7.5:
            reward -= 1.0  # pH too extreme
        
        # Reward for optimal water level (70-85%)
        if 70.0 <= self.water_level <= 85.0:
            reward += 0.3
        elif self.water_level < 30.0:
            reward -= 0.8  # Too dry
        elif self.water_level > 95.0:
            reward -= 0.5  # Too much water
        
        # Reward for optimal light (60-80%)
        if 60.0 <= self.light_intensity <= 80.0:
            reward += 0.2
        
        # Reward for plant growth
        avg_growth = np.mean(self.plant_grid)
        reward += avg_growth * 0.1
        
        # MASSIVE reward for actually harvesting plants (action-based)
        if self.last_harvest_count > 0:
            # Give HUGE reward per plant harvested to make it the primary goal
            reward += self.last_harvest_count * 50.0  # Increased to 50 per plant!
            # Big bonus for harvesting many plants at once (efficient farming)
            if self.last_harvest_count >= 10:
                reward += 100.0  # Extra bonus for bulk harvesting
        
        # STRONG PENALTY for NOT harvesting mature plants (opportunity cost)
        # This makes the agent WANT to harvest rather than hoard mature plants
        mature_count = np.sum(self.plant_grid >= 0.90)
        if mature_count > 0:
            # No reward for mature plants - only penalty for not harvesting them!
            if mature_count >= 5:  # If any significant number unharvested
                reward -= mature_count * 1.0  # Strong penalty per unharvested mature plant
            if mature_count >= 30:  # If many mature plants unharvested
                reward -= 50.0  # Extra penalty for wasting farm space
        
        # Small penalty for each step (encourage efficiency)
        reward -= 0.05
        
        # Bonus for maintaining optimal conditions
        if self._all_conditions_optimal():
            reward += 1.0
        
        return reward
    
    def _all_conditions_optimal(self) -> bool:
        """Check if all conditions are in optimal ranges."""
        return (1.5 <= self.ec <= 2.5 and 
                5.5 <= self.pH <= 6.5 and 
                70.0 <= self.water_level <= 85.0 and
                60.0 <= self.light_intensity <= 80.0)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if all plants are mature and harvested
        if np.all(self.plant_grid < 0.3):
            return True
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        mature_ratio = np.sum(self.plant_grid >= 0.90) / (self.grid_size * self.grid_size)
        avg_growth = np.mean(self.plant_grid)
        
        return np.array([
            self.ec,
            self.pH,
            self.water_level,
            self.light_intensity,
            avg_growth,
            mature_ratio,
            self.temperature,
            self.humidity,
            self.time_of_day
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        mature_count = int(np.sum(self.plant_grid >= 0.90))
        avg_growth = float(np.mean(self.plant_grid))
        
        return {
            "step": self.current_step,
            "ec": float(self.ec),
            "ph": float(self.pH),
            "water_level": float(self.water_level),
            "light_intensity": float(self.light_intensity),
            "mature_plants": mature_count,
            "avg_growth": avg_growth,
            "total_harvested": self.total_harvested,
            "optimal_conditions_ratio": self.optimal_conditions_count / max(self.current_step, 1)
        }
    
    def render(self):
        """Render the environment (delegated to rendering.py)."""
        if self.render_mode == "human":
            # Rendering is handled by the wrapper
            pass

