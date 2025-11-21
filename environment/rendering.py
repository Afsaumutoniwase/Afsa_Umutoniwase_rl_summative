"""
Advanced visualization for the Hydroponics Environment using Pygame.
Provides real-time visual feedback of the agent's state and actions.
"""

import pygame
import numpy as np
from typing import Optional, Tuple
import sys


class HydroponicsRenderer:
    """
    High-quality 2D visualization of the hydroponic farming environment.
    Shows grid, plant growth stages, system parameters, and agent actions.
    """
    
    def __init__(self, grid_size: int = 8, window_size: Tuple[int, int] = (1000, 800)):
        """
        Initialize the renderer.
        
        Args:
            grid_size: Size of the hydroponic grid
            window_size: Window dimensions (width, height)
        """
        self.grid_size = grid_size
        self.window_size = window_size
        self.cell_size = min(window_size[0] // (grid_size + 2), 
                            (window_size[1] - 200) // (grid_size + 2))
        
        pygame.init()
        self.screen = pygame.display.get_surface()
        if self.screen is None:
            self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("FarmSmart Rwanda - Hydroponics RL Environment")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.bold_font = pygame.font.Font(None, 32)
        
        # Color scheme
        self.colors = {
            'background': (240, 248, 255),  # Alice blue
            'grid': (200, 200, 200),
            'grid_dark': (150, 150, 150),
            'water': (135, 206, 250),  # Sky blue
            'plant_young': (144, 238, 144),  # Light green
            'plant_mature': (34, 139, 34),  # Forest green
            'plant_ready': (255, 215, 0),  # Gold
            'text': (0, 0, 0),
            'text_highlight': (0, 100, 0),
            'panel': (255, 255, 255),
            'border': (100, 100, 100),
            'optimal': (0, 200, 0),
            'warning': (255, 165, 0),
            'danger': (255, 0, 0)
        }
    
    def render(self, env_state: dict, action: Optional[int] = None, 
               reward: Optional[float] = None, step: int = 0):
        """
        Render the current state of the environment.
        
        Args:
            env_state: Dictionary containing environment state
            action: Last action taken (optional)
            reward: Last reward received (optional)
            step: Current step number
        """
        self.screen.fill(self.colors['background'])
        
        # Draw main grid
        grid_offset_x = 50
        grid_offset_y = 50
        self._draw_grid(env_state, grid_offset_x, grid_offset_y)
        
        # Draw information panels
        self._draw_info_panel(env_state, action, reward, step)
        
        # Draw parameter bars
        self._draw_parameter_bars(env_state)
        
        pygame.display.flip()
        self.clock.tick(4)  # Match environment's render_fps
    
    def _draw_grid(self, env_state: dict, offset_x: int, offset_y: int):
        """Draw the hydroponic grid with plants."""
        plant_grid = env_state.get('plant_grid', np.zeros((self.grid_size, self.grid_size)))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = offset_x + j * self.cell_size
                y = offset_y + i * self.cell_size
                
                # Draw cell background (water)
                cell_rect = pygame.Rect(x, y, self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(self.screen, self.colors['water'], cell_rect)
                pygame.draw.rect(self.screen, self.colors['grid'], cell_rect, 1)
                
                # Draw plant based on growth stage
                growth = plant_grid[i, j]
                if growth < 0.3:
                    # Young plant - small green circle
                    radius = int(self.cell_size * 0.2 * growth / 0.3)
                    if radius > 0:
                        pygame.draw.circle(self.screen, self.colors['plant_young'],
                                         (x + self.cell_size // 2, y + self.cell_size // 2),
                                         radius)
                elif growth < 0.95:
                    # Growing plant - medium green circle
                    radius = int(self.cell_size * 0.3)
                    pygame.draw.circle(self.screen, self.colors['plant_mature'],
                                     (x + self.cell_size // 2, y + self.cell_size // 2),
                                     radius)
                else:
                    # Mature plant - gold circle (ready to harvest)
                    radius = int(self.cell_size * 0.35)
                    pygame.draw.circle(self.screen, self.colors['plant_ready'],
                                     (x + self.cell_size // 2, y + self.cell_size // 2),
                                     radius)
                    # Add harvest indicator
                    pygame.draw.circle(self.screen, self.colors['text'],
                                     (x + self.cell_size // 2, y + self.cell_size // 2),
                                     radius, 2)
        
        # Draw grid label
        label = self.bold_font.render("Hydroponic Grid", True, self.colors['text'])
        self.screen.blit(label, (offset_x, offset_y - 30))
    
    def _draw_info_panel(self, env_state: dict, action: Optional[int], 
                        reward: Optional[float], step: int):
        """Draw information panel on the right side."""
        panel_x = 50 + self.grid_size * self.cell_size + 20
        panel_y = 50
        panel_width = 300
        panel_height = 600
        
        # Panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, 2)
        
        y_offset = panel_y + 20
        
        # Title
        title = self.bold_font.render("System Status", True, self.colors['text_highlight'])
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 40
        
        # Step counter
        step_text = self.font.render(f"Step: {step}", True, self.colors['text'])
        self.screen.blit(step_text, (panel_x + 10, y_offset))
        y_offset += 30
        
        # Last action
        if action is not None:
            action_names = [
                "Increase Nutrients", "Decrease Nutrients", "Increase pH",
                "Decrease pH", "Add Water", "Increase Light", "Decrease Light",
                "Harvest", "Wait"
            ]
            action_text = self.font.render(f"Action: {action_names[action]}", 
                                         True, self.colors['text'])
            self.screen.blit(action_text, (panel_x + 10, y_offset))
            y_offset += 30
        
        # Last reward
        if reward is not None:
            color = self.colors['optimal'] if reward > 0 else self.colors['danger']
            reward_text = self.font.render(f"Reward: {reward:.2f}", True, color)
            self.screen.blit(reward_text, (panel_x + 10, y_offset))
            y_offset += 40
        
        # System parameters
        params = [
            ("EC", env_state.get('ec', 0), 1.5, 2.5),
            ("pH", env_state.get('ph', 0), 5.5, 6.5),
            ("Water %", env_state.get('water_level', 0), 70, 85),
            ("Light %", env_state.get('light_intensity', 0), 60, 80),
            ("Temp °C", env_state.get('temperature', 0), 22, 26),
            ("Humidity %", env_state.get('humidity', 0), 50, 70),
        ]
        
        for param_name, value, opt_min, opt_max in params:
            param_label = self.small_font.render(f"{param_name}:", True, self.colors['text'])
            self.screen.blit(param_label, (panel_x + 10, y_offset))
            
            # Determine color based on optimal range
            if opt_min <= value <= opt_max:
                color = self.colors['optimal']
            elif abs(value - (opt_min + opt_max) / 2) < (opt_max - opt_min):
                color = self.colors['warning']
            else:
                color = self.colors['danger']
            
            value_text = self.small_font.render(f"{value:.2f}", True, color)
            self.screen.blit(value_text, (panel_x + 150, y_offset))
            y_offset += 25
        
        y_offset += 20
        
        # Plant statistics
        mature_count = env_state.get('mature_plants', 0)
        avg_growth = env_state.get('avg_growth', 0)
        total_harvested = env_state.get('total_harvested', 0)
        
        stats_title = self.font.render("Plant Statistics", True, self.colors['text_highlight'])
        self.screen.blit(stats_title, (panel_x + 10, y_offset))
        y_offset += 30
        
        mature_text = self.small_font.render(f"Mature Plants: {mature_count}", 
                                           True, self.colors['text'])
        self.screen.blit(mature_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        growth_text = self.small_font.render(f"Avg Growth: {avg_growth:.2%}", 
                                            True, self.colors['text'])
        self.screen.blit(growth_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        harvest_text = self.small_font.render(f"Total Harvested: {total_harvested}", 
                                             True, self.colors['optimal'])
        self.screen.blit(harvest_text, (panel_x + 10, y_offset))
    
    def _draw_parameter_bars(self, env_state: dict):
        """Draw visual parameter bars at the bottom."""
        bar_y = 50 + self.grid_size * self.cell_size + 20
        bar_height = 30
        bar_width = 150
        spacing = 20
        
        params = [
            ("EC", env_state.get('ec', 0), 0, 4, 1.5, 2.5),
            ("pH", env_state.get('ph', 0), 4, 8, 5.5, 6.5),
            ("Water", env_state.get('water_level', 0), 0, 100, 70, 85),
            ("Light", env_state.get('light_intensity', 0), 0, 100, 60, 80),
        ]
        
        for i, (name, value, min_val, max_val, opt_min, opt_max) in enumerate(params):
            x = 50 + i * (bar_width + spacing)
            
            # Label
            label = self.small_font.render(name, True, self.colors['text'])
            self.screen.blit(label, (x, bar_y - 20))
            
            # Bar background
            bar_rect = pygame.Rect(x, bar_y, bar_width, bar_height)
            pygame.draw.rect(self.screen, (220, 220, 220), bar_rect)
            pygame.draw.rect(self.screen, self.colors['border'], bar_rect, 1)
            
            # Optimal range indicator
            opt_start = int((opt_min - min_val) / (max_val - min_val) * bar_width)
            opt_end = int((opt_max - min_val) / (max_val - min_val) * bar_width)
            opt_rect = pygame.Rect(x + opt_start, bar_y, opt_end - opt_start, bar_height)
            pygame.draw.rect(self.screen, (200, 255, 200), opt_rect)
            
            # Current value indicator
            value_pos = int((value - min_val) / (max_val - min_val) * bar_width)
            value_pos = max(0, min(bar_width, value_pos))
            
            # Color based on position
            if opt_min <= value <= opt_max:
                color = self.colors['optimal']
            else:
                color = self.colors['warning']
            
            pygame.draw.line(self.screen, color, 
                           (x + value_pos, bar_y), 
                           (x + value_pos, bar_y + bar_height), 3)
    
    def close(self):
        """Close the renderer and cleanup."""
        pygame.quit()
    
    def save_frame(self, filename: str, env_state: dict, action: Optional[int] = None,
                   reward: Optional[float] = None, step: int = 0):
        """Save current frame as an image."""
        self.render(env_state, action, reward, step)
        pygame.image.save(self.screen, filename)


def create_env_wrapper(env, render_mode='human'):
    """
    Create a wrapper that integrates the renderer with the environment.
    Note: This is a helper function. Rendering is typically handled
    directly in main.py for better control.
    """
    if render_mode == 'human':
        # Renderer will be created and used in main.py
        pass
    return env

