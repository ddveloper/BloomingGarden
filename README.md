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


### CrazyGames bot integration
You can now run a browser bot against the new host:

```bash
python3 web_bot_crazygames.py --steps 500 --beam-width 10 --depth 2 --samples 4
```

Recommended setup flow:
1. Install deps: `pip install -r requirements.txt` and `playwright install chromium`.
2. Capture a fresh game-frame screenshot for calibration:
   ```bash
   python3 calibrate_crazygames.py --out results/crazygames_frame.png --wait-ms 60000
   ```
3. Update `crazygames_config.json` with accurate board/next-flower coordinates for your screen layout.
4. Start the page and let the bot take over.

Notes:
- The game runs in an iframe/canvas and UI layout may change; coordinate calibration is expected.
- The bot uses `baseline_agent.py` planner directly for decision making.


Troubleshooting:
- If you see `AttributeError: 'Frame' object has no attribute 'screenshot'`, pull latest changes. The scripts now capture the iframe using `page.screenshot(clip=...)`, which is compatible with Playwright Python.
- On Windows, if `playwright` command is not found, run `python -m playwright install chromium` instead.

- If calibration cannot resolve iframe bounds (ads/overlay/frame recreation), it now saves a full-page fallback screenshot instead of crashing.
