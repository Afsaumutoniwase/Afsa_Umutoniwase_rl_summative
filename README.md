# FarmSmart Rwanda: AI-Powered Hydroponics RL System

**Student:** Afsa Umutoniwase  
**Course:** Machine Learning Techniques II - Reinforcement Learning  
**Institution:** African Leadership University (ALU)  
**Mission:** Revolutionize agriculture in Rwanda through AI-powered hydroponic farming systems

---

## Project Overview

This project implements and compares four reinforcement learning algorithms to autonomously manage a hydroponic farming system. The AI agent learns to optimize critical growing parameters—nutrient concentration (EC), pH levels, water supply, and lighting—while maximizing crop yield through timely harvesting in a simulated 8×8 hydroponic grid with 64 plant slots.

**Mission Context:** Rwanda faces agricultural challenges including limited arable land (45% of land is arable but highly fragmented), climate variability, and knowledge gaps in modern farming techniques. This project addresses these challenges by developing an AI system that can autonomously manage resource-efficient hydroponic farms.

## Project Structure

```
Afsa_Umutoniwase_rl_summative/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py            # Custom Gymnasium HydroponicsEnv
│   └── rendering.py             # Pygame 2D visualization system
│
├── training/
│   ├── __init__.py
│   ├── dqn_training.py          # DQN training with hyperparameter tuning
│   └── pg_training.py           # Policy Gradient (PPO, A2C, REINFORCE) training
│
├── models/
│   ├── dqn/
│   │   └── best/
│   │       └── best_model.zip   # Trained DQN model
│   └── pg/
│       ├── ppo/best/            # Trained PPO model
│       ├── a2c/best/            # Trained A2C model
│       └── reinforce/best/      # Trained REINFORCE model
│
├── results/
│   ├── dqn_tuning/
│   │   └── tuning_results.json  # 12 hyperparameter configurations
│   ├── ppo_tuning/
│   │   └── tuning_results.json  # 12 hyperparameter configurations
│   ├── a2c_tuning/
│   │   └── tuning_results.json  # 12 hyperparameter configurations
│   ├── reinforce_tuning/
│   │   └── tuning_results.json  # 12 hyperparameter configurations
│   ├── *_final/logs/            # Final training logs and evaluations
│   ├── hyperparameter_comparison.png
│   ├── final_model_comparison.png
│   ├── learning_curves.png
│   ├── performance_metrics.png
│   └── performance_summary_table.png
│
├── static_random_agent/         # Random agent visualization frames
│   ├── frame_0000.png
│   ├── frame_0005.png
│   └── ...
│
├── main.py                      # Entry point - run trained models
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── VIDEO_RECORDING_GUIDE.md     # Video demonstration script
```

## Environment Details

### Observation Space (9-dimensional continuous)
| Parameter | Range | Optimal | Description |
|-----------|-------|---------|-------------|
| **EC (Electrical Conductivity)** | 0.0 - 4.0 | 1.5 - 2.5 | Nutrient concentration |
| **pH Level** | 4.0 - 8.0 | 5.5 - 6.5 | Acidity/alkalinity |
| **Water Level** | 0 - 100% | 70 - 85% | Tank capacity percentage |
| **Light Intensity** | 0 - 100% | 60 - 80% | Lighting percentage |
| **Average Growth Stage** | 0.0 - 1.0 | → 0.90+ | Mean plant maturity |
| **Mature Plants Ratio** | 0.0 - 1.0 | → 1.0 | Harvestable plants fraction |
| **Temperature** | 15 - 35°C | 22 - 26°C | Ambient temperature |
| **Humidity** | 30 - 90% | 50 - 70% | Relative humidity |
| **Time of Day** | 0.0 - 1.0 | N/A | Normalized daily cycle |

### Action Space (9 discrete actions)
| Action | Effect | Magnitude |
|--------|--------|-----------|
| **0** | Increase nutrients (EC) | +0.2 units |
| **1** | Decrease nutrients (EC) | -0.2 units |
| **2** | Increase pH | +0.1 units |
| **3** | Decrease pH | -0.1 units |
| **4** | Add water | +5% level |
| **5** | Increase light | +10% intensity |
| **6** | Decrease light | -10% intensity |
| **7** | Harvest mature plants | Harvests all plants ≥ 0.90 growth |
| **8** | Do nothing | Wait and observe |

### Reward Structure
**Positive Rewards:**
- +0.5 per step for optimal EC (1.5-2.5)
- +0.5 per step for optimal pH (5.5-6.5)
- +0.3 per step for optimal water (70-85%)
- +0.2 per step for optimal light (60-80%)
- +50 **per plant harvested**
- +100 bonus for bulk harvest (≥10 plants)
- +1.0 per step when all conditions optimal

**Negative Rewards:**
- -1.0 per step for severe EC imbalance
- -1.0 per step for extreme pH
- -0.8 per step for low water (<30%)
- -1.0 per unharvested mature plant
- -50 penalty if ≥30 plants unharvested
- -0.05 step penalty (efficiency)

**Maximum Possible:** ~3,300 reward per episode (64 plants × 50 + bonuses)

### Episode Termination
- Maximum 200 steps per episode
- Early termination possible if all plants harvested and system reset

## Quick Start

### Prerequisites
- Python 3.8+ (tested on Python 3.12.6)
- pip package manager
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Afsaumutoniwase/Afsa_Umutoniwase_rl_summative.git
cd Afsa_Umutoniwase_rl_summative
```

2. **Create virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `gymnasium` - RL environment framework
- `stable-baselines3[extra]` - RL algorithms (DQN, PPO, A2C)
- `pygame` - 2D visualization
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `tensorboard` - Training monitoring

## Usage

### Option 1: Run Pre-Trained Models (Recommended)

The repository includes fully trained models ready to demonstrate.

**Run best performing agent (REINFORCE):**
```bash
python main.py --mode trained --algorithm reinforce --episodes 3
```

**Run other algorithms:**
```bash
# Deep Q-Network (Value-Based)
python main.py --mode trained --algorithm dqn --episodes 3

# Proximal Policy Optimization
python main.py --mode trained --algorithm ppo --episodes 3

# Advantage Actor-Critic
python main.py --mode trained --algorithm a2c --episodes 3
```

**Compare all models:**
```bash
python main.py --mode compare --episodes 3
```

**Random agent baseline (no model):**
```bash
python main.py --mode random --episodes 1
```
This creates visualization frames in `static_random_agent/` folder.

---

### Option 2: Train Models from Scratch

**⚠️ Warning:** Training takes ~1-3 hours per algorithm with hyperparameter tuning.

**Train DQN (Value-Based):**
```bash
python training/dqn_training.py
```
- Tests 12 hyperparameter configurations
- Each config: 200,000 timesteps
- Saves best model to `models/dqn/best/`

**Train Policy Gradient Methods:**
```bash
python training/pg_training.py
```
- Trains PPO, A2C, and REINFORCE
- 12 configs × 3 algorithms = 36 runs
- Each config: 200,000 timesteps
- Total time: ~6-8 hours

**Results saved to:**
- `results/{algorithm}_tuning/tuning_results.json`
- `results/{algorithm}_final/logs/`

## Algorithms Implemented

### 1. DQN (Deep Q-Network) - Value-Based
- **Type:** Off-policy, value-based
- **Architecture:** MLP with 256×256 hidden layers
- **Key Features:** Experience replay, target network, epsilon-greedy exploration
- **Best Config:** Learning rate 0.0001, gamma 0.995, buffer size 100k
- **Performance:** Mean reward 2,948.55 ± 1,544.62
- **Strengths:** Sample efficient, learns from past experiences
- **Weaknesses:** High variance, struggles with sparse rewards

### 2. PPO (Proximal Policy Optimization) - Policy Gradient
- **Type:** On-policy, actor-critic
- **Architecture:** Shared 256×256 MLP with separate policy/value heads
- **Key Features:** Clipped surrogate objective, GAE, entropy bonus
- **Best Config:** Learning rate 0.0003, 2048 steps, 64 batch size, 20 epochs
- **Performance:** Mean reward 2,944.81 ± 619.75
- **Strengths:** Stable training, good for continuous control
- **Weaknesses:** Conservative updates, slower convergence

### 3. A2C (Advantage Actor-Critic) - Policy Gradient
- **Type:** On-policy, actor-critic
- **Architecture:** 256×256 MLP with advantage estimation
- **Key Features:** Synchronous updates, 5-step returns, value function baseline
- **Best Config:** Learning rate 0.0005, n_steps 5, GAE lambda 1.0
- **Performance:** Mean reward 3,651.89 ± 381.03
- **Strengths:** Fast convergence, balanced bias-variance
- **Weaknesses:** Requires careful tuning

### 4. REINFORCE - Policy Gradient
- **Type:** On-policy, Monte Carlo policy gradient
- **Architecture:** 256×256 MLP policy network
- **Key Features:** Complete episode returns, no variance reduction
- **Best Config:** Learning rate 0.0005, 200 steps, GAE lambda 1.0
- **Performance:** Mean reward 3,851.95 ± 285.04 (BEST)
- **Strengths:** Simple, stable, best for episodic tasks
- **Weaknesses:** High sample complexity, slow initial learning

---

## Hyperparameter Tuning

**Extensive tuning performed:** 12 configurations per algorithm = **48 total training runs**

**Parameters tuned:**
- Learning rates: 1×10⁻⁴ to 5×10⁻³
- Network architectures: [64,64] to [512,512]
- Batch sizes: 32 to 512
- Training frequencies: 1 to 20 epochs
- Discount factors (gamma): 0.98 to 0.995
- GAE lambda: 0.9 to 1.0
- Exploration strategies (DQN): epsilon decay schedules
- Clip ranges (PPO): 0.1 to 0.3

**Results:**
- All tuning results: `results/{algorithm}_tuning/tuning_results.json`
- Best configurations identified and used for final training
- Visualizations: `results/hyperparameter_comparison.png`

## Visualization

### Pygame 2D Rendering System
The environment features professional visualization with:

**Plant Grid:**
- 8×8 grid representing 64 hydroponic plant slots
- Color-coded growth stages:
  - Red/Dark: Young plants (0.0 - 0.3)
  - Yellow: Growing plants (0.3 - 0.7)
  - Green: Mature plants (0.7 - 0.9)
  - Bright Green: Harvestable (≥ 0.9)

**Metrics Dashboard:**
- Real-time parameter displays with visual bars
- Optimal range indicators (green zones)
- Current values for EC, pH, water, light, temperature, humidity
- Episode statistics (steps, total reward, harvest count)
- Current action and immediate reward feedback

**Demo Modes:**
- Real-time visualization at 4 FPS
- Frame-by-frame saving for analysis
- Random agent static frames in `static_random_agent/`

---

## Performance Results

### Final Model Comparison (200,000 timesteps each)

| Algorithm | Mean Reward | Std Dev | Stability | Harvests/Episode |
|-----------|-------------|---------|-----------|------------------|
| **REINFORCE** | **3,851.95** | ±285.04 | Excellent | ~480-500 |
| **A2C** | 3,651.89 | ±381.03 | Very Good | ~460-480 |
| **PPO** | 2,944.81 | ±619.75 | Good | ~380-420 |
| **DQN** | 2,948.55 | ±1,544.62 | Fair | ~350-450 |

**Key Findings:**
- **REINFORCE** performs best due to complete episode returns matching sparse harvest rewards
- **A2C** balances speed and performance with advantage estimation
- **PPO** conservative updates limit peak performance
- **DQN** struggles with high variance and sparse rewards

**Convergence Times:**
- DQN: 400-500 episodes
- PPO: 600-700 episodes
- A2C: 500-600 episodes
- REINFORCE: 700-800 episodes (slower but more stable)

**Visualizations available in `results/` folder:**
- Hyperparameter comparison charts
- Learning curves over time
- Performance metrics comparison
- Statistical summary tables

## Mission Alignment: FarmSmart Rwanda

This project directly addresses Rwanda's agricultural challenges:

### The Problem
- **Limited arable land** (only 45% usable, highly fragmented)
- **Climate variability** affecting traditional farming
- **Knowledge gaps** in modern agricultural techniques
- **Resource constraints** limiting farming innovation
- **Food security** challenges with growing population

### Our Solution
- **AI-powered management** eliminates need for expert knowledge
- **Hydroponic systems** maximize yield in limited space (no soil required)
- **Data-driven optimization** adapts to local conditions
- **Resource efficiency** through intelligent water/nutrient management
- **Scalable technology** from small farms to commercial operations

### Impact
- Makes advanced agriculture accessible to Rwandan farmers
- Reduces dependency on traditional farming constraints
- Provides foundation for smart agriculture infrastructure
- Demonstrates ML/AI application in real-world African context

---

## Technical Report

Complete project analysis available in:
- `Machine_Learning_Techniques_II - Summative_Assignment - Report.txt`
- Includes detailed methodology, results, and discussion
- See `VIDEO_RECORDING_GUIDE.md` for demonstration script

---

## Repository

**GitHub:** [github.com/Afsaumutoniwase/Afsa_Umutoniwase_rl_summative](https://github.com/Afsaumutoniwase/Afsa_Umutoniwase_rl_summative)

---

## License

This project is part of an academic assignment for ALU's BSE program in Machine Learning Techniques II.

---

## Contact

**Student:** Afsa Umutoniwase  
**Institution:** African Leadership University (ALU)  
**Email:** a.umutoniwa@alustudent.com  
**GitHub:** [@Afsaumutoniwase](https://github.com/Afsaumutoniwase)

---

## Acknowledgments

- **Stable-Baselines3** team for excellent RL library
- **Gymnasium** for environment framework
- **Pygame** community for visualization tools
- ALU ML Techniques II course instructors

---
