## BloomingGarden
Solve the Bloomin' Garden puzzle game with CV + RL.

> Original source in this repo points to an old Miniclip page (`https://www.miniclip.com/games/bloomin-gardens/en/`), which may no longer host a playable build.

### Current project direction
To keep progress independent of third-party hosting, this repo supports local agent iteration directly on the Gym environment in `gym-bloomingGarden/`.

### Strong local baseline (high-score planner)
Run the planner:
```bash
python3 baseline_agent.py --episodes 20 --seed 123 --beam-width 12 --depth 2 --samples 5
```

What it does:
- Generates all legal moves with BFS reachability.
- Scores tactical value (immediate clear, mobility, board space, chain potential).
- Uses stochastic lookahead by sampling random flower placement outcomes.

This gives a much better starting point for reaching high scores and generating trajectories for ML training.

### If you still want to recover a browser-playable build
1. Search web archives (Internet Archive / mirrors) for `bloomin-gardens` flash assets.
2. If you recover a SWF/HTML bundle, run it locally with a Flash-compatible runtime (e.g. Ruffle).
3. Point CV capture logic in `detect.py` to your local window coordinates.

### Next ML steps toward 3000+
- Use this planner to generate imitation-learning trajectories (state, action, return).
- Train a policy/value network on those trajectories, then fine-tune with RL.
- Evaluate with fixed seeds and track: min / median / avg / max across episodes.
- Promote models only when median crosses your target threshold (e.g. 3000+).
