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
python3 web_bot_crazygames.py --steps 500 --beam-width 10 --depth 2 --samples 4 --preview-action-ms 5000 --drag-move-steps 18 --drag-hover-ms 300 --debug-capture-steps 3
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
- Runtime now prints decision logs per step (empty cells, coming flowers, action count, chosen move, top-3 scored candidates).
- Runtime now waits for CLI confirmation before acting: type `y` + Enter in terminal to start detect/action loop.
- Runtime highlights bot clicks on-screen (green = source, red = destination) so interactions are visually traceable.
- Flower movement now follows a 3 sub-step interaction model: click source -> move cursor to target -> click (optionally confirm click) at target.
- Before each move, bot draws board border debug lines (red) and planned source/destination markers (red/yellow) for visibility.
- Board mapping is perspective-aware via `board_quad` (lt, lb, rt, rb) in `crazygames_config.json`; tune these 4 points to match trapezoid board corners.
- Overlay and click targets are now projected from homography (not flat `cell_size` grid), so far-side/near-side distortion is handled.
- Runtime prints per-step timing breakdown with timestamps (`capture`, `decide`, `overlay+preview`, `action`, `think`, `total`) to show where time is spent.
- Bot auto-saves screenshots for first N steps (default 3) while red debug overlay is shown, for easier coordinate tuning.


Troubleshooting:
- If you see `AttributeError: 'Frame' object has no attribute 'screenshot'`, pull latest changes. The scripts now capture the iframe using `page.screenshot(clip=...)`, which is compatible with Playwright Python.
- On Windows, if `playwright` command is not found, run `python -m playwright install chromium` instead.

- If calibration cannot resolve iframe bounds (ads/overlay/frame recreation), it now saves a full-page fallback screenshot instead of crashing.

Example without visual markers:
```bash
python3 web_bot_crazygames.py --steps 500 --wait-ms 60000 --hide-click-overlay
```

If your game variant needs only one target click, disable confirm click:
```bash
python3 web_bot_crazygames.py --steps 500 --wait-ms 60000 --no-confirm-target-click
```

Startup control:
- After launching bot, it waits in terminal for: `Type y then Enter to start bot actions:`



How to tune `board_quad` (lt, lb, rt, rb):
1. Run bot with preview enabled so projected red border/grid is visible.
2. Edit `crazygames_config.json` and move points one by one:
   - `lt`: top-left board corner
   - `lb`: bottom-left board corner
   - `rt`: top-right board corner
   - `rb`: bottom-right board corner
3. If overlay is shifted right/left, move all x values together.
4. If overlay is shifted up/down, move all y values together.
5. If far edge is too wide/narrow, adjust `lt/rt` x values.
6. If near edge is too wide/narrow, adjust `lb/rb` x values.
7. Re-run and iterate until red projected grid lines sit on real cell seams.

Debug tips:
- Overlay now draws detected flower labels (cell value IDs) in cyan on occupied cells.
- Terminal logs `detected-coming=...` and full detected flower list before each action.

Debug screenshot options:
```bash
python3 web_bot_crazygames.py --debug-capture-steps 3 --debug-capture-dir results/debug_steps
```
