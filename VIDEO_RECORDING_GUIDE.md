# VIDEO RECORDING GUIDE - RL Summative Assignment

**Duration:** 3 minutes maximum  
**Requirements:** Full screen share + Camera ON  
**Worth:** 10 points (20% of total grade)

---

## PRE-RECORDING CHECKLIST

### Setup (5 minutes before recording):
- [ ] Close unnecessary applications
- [ ] Clear desktop clutter
- [ ] Test camera and microphone
- [ ] Open required windows:
  - [ ] Terminal/PowerShell (full screen capable)
  - [ ] Project folder ready to navigate
  - [ ] This script open for reference
- [ ] Run a test simulation to ensure it works:
  ```bash
  python main.py --mode trained --algorithm reinforce --episodes 3
  ```

### Recording Tools:
- **Windows:** Xbox Game Bar (Win + G) or OBS Studio
- **Zoom:** Record meeting (share screen + enable camera)
- **PowerPoint:** Built-in screen recording with webcam

---

## VIDEO SCRIPT (3 minutes)

### **INTRODUCTION (30 seconds)**
*Camera: ON, Screen: Show project folder*

> "Hello, I'm Afsa Umutoniwase, and this is my Reinforcement Learning Summative Assignment. 
>
> **The Problem:** Rwanda faces agricultural challenges including limited arable land, climate variability, and knowledge gaps in modern farming. Traditional farming cannot meet the growing food demand.
>
> **My Solution:** FarmSmart Rwanda - an AI-powered hydroponic system that uses reinforcement learning to autonomously manage nutrient levels, pH, water, and lighting to maximize crop yield in resource-constrained environments."

---

### **AGENT BEHAVIOR & OBJECTIVE (45 seconds)**
*Screen: Show report or environment code briefly*

> "**Agent Behavior:** The agent acts as an autonomous farm manager. It observes 9 environmental parameters including EC (nutrient concentration), pH, water level, light intensity, and plant growth stages. It can take 9 actions: adjusting nutrients up or down, pH up or down, adding water, controlling light, harvesting mature plants, or waiting.
>
> **Agent Objective:** Maximize crop yield by maintaining optimal growing conditions and harvesting plants at peak maturity. The agent must balance short-term parameter optimization with long-term harvest rewards.
>
> **Reward Structure:** The agent receives +50 points per plant harvested, with bulk bonuses for harvesting 10+ plants. It gets positive rewards for maintaining optimal EC (1.5-2.5), pH (5.5-6.5), water (70-85%), and light (60-80%). Penalties are applied for extreme conditions and for leaving mature plants unharvested, which encourages timely harvesting."

---

### **RUN SIMULATION (90 seconds)**
*Screen: Full terminal + GUI visible*

> "Now let me demonstrate the best-performing agent - REINFORCE - which achieved a mean reward of 3,851.95 with excellent stability. Watch both the GUI visualization and terminal output."

**Run command:**
```bash
cd c:\Users\HP\Desktop\ALU\Afsa_Umutoniwase_rl_summative
python main.py --mode trained --algorithm reinforce --episodes 3
```

**While running, narrate (naturally, not scripted):**
- Point out the plant grid showing growth stages (green colors)
- Mention the metrics dashboard (EC, pH, water, light)
- Highlight when the agent harvests (action 7)
- Note the reward accumulation in terminal
- Point out episode completion and statistics

---

### **PERFORMANCE EXPLANATION (15 seconds)**
*Screen: Keep showing final statistics or briefly show graphs*

> "As you can see, the trained agent successfully harvests 60+ plants per episode, maintains optimal conditions, and achieves rewards above 3,500. The agent learned to time harvests perfectly and balance resource management.
>
> I trained four algorithms - DQN, PPO, A2C, and REINFORCE. REINFORCE performed best due to its complete episode returns matching the long-horizon nature of plant growth and sparse harvesting rewards. All results, hyperparameters, and analysis are documented in my report.
>
> Thank you!"

---

## RUBRIC REQUIREMENTS CHECKLIST

Ensure your video includes ALL of these for full 10 points:

- [ ] **Full screen share** (entire screen visible, not just window)
- [ ] **Camera ON** (face visible throughout)
- [ ] **Problem statement** (Rwanda agriculture, FarmSmart mission)
- [ ] **Agent behavior** (9 actions, 9 observations)
- [ ] **Reward structure** (+50/harvest, optimal conditions, penalties)
- [ ] **Agent objective** (maximize yield, optimal harvesting)
- [ ] **GUI visualization** (Pygame showing plants, metrics)
- [ ] **Terminal outputs** (verbose episode stats, rewards)
- [ ] **Performance explanation** (REINFORCE best, 3,851 mean reward)
- [ ] **Duration:** Under 3 minutes

---

## AFTER RECORDING

### Upload Options:

**Option 1: YouTube (Recommended)**
1. Upload as "Unlisted" (not private, not public)
2. Title: "Afsa Umutoniwase - RL Summative - FarmSmart Rwanda"
3. Copy the share link
4. Test link in incognito browser to verify accessibility

**Option 2: Google Drive**
1. Upload video file
2. Right-click → Share → "Anyone with the link can view"
3. Copy link
4. Test link in incognito browser

**Option 3: OneDrive/Dropbox**
1. Upload video
2. Create shareable link (view permissions)
3. Test link

### Add Link to Report:
Replace the placeholder in your report PDF:
```
Video Recording: [YOUR_LINK_HERE]
```

---

## TIPS FOR SUCCESS

### DO:
- Practice the script 2-3 times before recording
- Speak clearly and at moderate pace
- Show enthusiasm for your project
- Point with cursor to highlight important elements
- Keep camera steady (laptop on stable surface)
- Ensure good lighting on your face
- Test the trained model works BEFORE recording

### DON'T:
- Read from script robotically (sound natural)
- Go over 3 minutes (will lose points)
- Forget to start screen recording
- Hide camera or use virtual background
- Show only terminal OR only GUI (show both)
- Skip the performance explanation
- Use a model that isn't trained or fails

---

## RECORDING WORKFLOW

1. **Prepare** (10 min)
   - Test model run
   - Clear screen
   - Review script

2. **Record** (5-10 min including retakes)
   - Start camera + screen recording
   - Follow script naturally
   - Run simulation
   - Stop recording

3. **Review** (5 min)
   - Watch full video
   - Check all rubric items
   - Verify audio/video quality
   - Re-record if needed

4. **Upload** (10 min)
   - Upload to platform
   - Set permissions correctly
   - Test link
   - Add to report

---

## COMMON MISTAKES TO AVOID

1. **Recording only 1 episode** - Run 3 episodes for better demonstration
2. **Not showing full screen** - Rubric specifically requires "share entire screen"
3. **Camera off** - Automatic point deduction
4. **Too long** - Over 3 minutes may not be viewed completely
5. **No terminal output** - Must show verbose statistics
6. **No GUI** - Must show Pygame visualization
7. **Wrong algorithm** - Use REINFORCE (best performance)
8. **Private video** - Professor cannot access

---

## EXPECTED OUTPUT

When you run the command, you should see:

**Terminal:**
```
Loading best REINFORCE model...
Episode 1/3 - Reward: 3847.25, Steps: 156, Harvests: 62
Episode 2/3 - Reward: 3921.50, Steps: 148, Harvests: 64
Episode 3/3 - Reward: 3786.75, Steps: 161, Harvests: 60
Average Reward: 3851.83 ± 67.42
```

**GUI:**
- 8x8 grid with plants changing colors
- Metrics dashboard showing EC, pH, water, light
- Harvest counter incrementing
- Episode statistics

---

**Good luck! This video is worth 20% of your grade - take the time to do it well!**
