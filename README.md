## BloomingGarden
Solve the Bloomin' Garden puzzle game with CV + RL.

> Original source in this repo points to an old Miniclip page (`https://www.miniclip.com/games/bloomin-gardens/en/`), which may no longer host a playable build.

### Current project direction
To keep progress independent of third-party hosting, this repo now supports local agent iteration directly on the Gym environment in `gym-bloomingGarden/`.

### Quick start (local simulation, no web host required)
```bash
python baseline_agent.py --episodes 20 --seed 123
```

This runs a deterministic heuristic baseline and prints min/avg/max scores. It gives a reproducible baseline while we build stronger ML models targeting 3000+ scores.

### If you still want to recover a browser-playable build
1. Search web archives (Internet Archive / mirrors) for `bloomin-gardens` flash assets.
2. If you recover a SWF/HTML bundle, run it locally with a Flash-compatible runtime (e.g. Ruffle).
3. Point CV capture logic in `detect.py` to your local window coordinates.

### Next ML steps
- Replace heuristic action ranking with policy/value learning on top of the same move generator.
- Use self-play rollouts to produce supervised warm-start data.
- Add evaluation gates (e.g. median score over N seeds) and require passing 3000+ before promoting models.
