"""Play Blooming Garden on CrazyGames using the local planner.

Workflow:
1) Open CrazyGames page and click into the game.
2) Capture board/next-flower state from the game iframe screenshot.
3) Use `HighScorePlanner` from `baseline_agent.py` to choose actions.
4) Click source/target cells to play automatically.

This script is intentionally config-driven because iframe/canvas layouts can change.
Update `crazygames_config.json` when the site UI shifts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import cv2
import numpy as np
from playwright.sync_api import Frame, sync_playwright

from baseline_agent import HighScorePlanner, PlannerConfig

EMPTY = 3


@dataclass
class BotConfig:
    url: str
    board_top_left: Tuple[int, int]
    cell_size: int
    next_slots: List[Tuple[int, int, int, int]]
    click_delay_s: float
    think_delay_s: float
    start_selectors: List[str]

    @staticmethod
    def load(path: Path) -> "BotConfig":
        raw = json.loads(path.read_text())
        return BotConfig(
            url=raw["url"],
            board_top_left=tuple(raw["board_top_left"]),
            cell_size=int(raw["cell_size"]),
            next_slots=[tuple(x) for x in raw["next_slots"]],
            click_delay_s=float(raw.get("click_delay_s", 0.08)),
            think_delay_s=float(raw.get("think_delay_s", 0.2)),
            start_selectors=list(raw.get("start_selectors", [])),
        )


class CrazyGamesBoardDetector:
    def __init__(self, cfg: BotConfig, centers_file: Path):
        self.cfg = cfg
        self.centers = np.loadtxt(centers_file, dtype=np.float32)
        if self.centers.shape != (8, 2):
            raise ValueError("centers.txt should contain 8x2 cluster centers")

    def detect(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        board = np.full((9, 9), EMPTY, dtype=int)
        ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCR_CB)
        x0, y0 = self.cfg.board_top_left
        step = self.cfg.cell_size

        # sample center patch per cell
        for r in range(9):
            for c in range(9):
                cx = x0 + c * step + step // 2
                cy = y0 + r * step + step // 2
                patch = ycc[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5, 1:]
                feat = np.mean(patch, axis=(0, 1))
                board[r][c] = self._nearest_flower(feat)

        next_flowers: List[int] = []
        for x, y, w, h in self.cfg.next_slots:
            patch = ycc[y : y + h, x : x + w, 1:]
            feat = np.mean(patch, axis=(0, 1))
            next_flowers.append(self._nearest_flower(feat))

        return board, next_flowers

    def _nearest_flower(self, feat: np.ndarray) -> int:
        dists = np.linalg.norm(self.centers - feat[None, :], axis=1)
        idx = int(np.argmin(dists))
        return idx


class CrazyGamesBot:
    def __init__(self, config: BotConfig, centers_path: Path, planner_cfg: PlannerConfig):
        self.config = config
        self.detector = CrazyGamesBoardDetector(config, centers_path)
        self.planner = HighScorePlanner(planner_cfg)

    def run(self, steps: int = 500, headless: bool = False) -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(self.config.url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            frame = self._resolve_game_frame(page)
            self._click_start_buttons(page, frame)

            for step in range(steps):
                board, coming = self._capture_state(frame)
                env_like = SimpleNamespace(brd=board, coming=coming, score=0)
                action = self.planner.choose_action(env_like)
                self._play_action(frame, action)
                print(f"step={step} action={action}")
                time.sleep(self.config.think_delay_s)

            browser.close()

    def _resolve_game_frame(self, page) -> Frame:
        page.wait_for_timeout(4000)
        for fr in page.frames:
            name = (fr.name or "").lower()
            url = (fr.url or "").lower()
            if "crazygames" in url and ("gameframe" in name or "game" in url):
                return fr
        # fallback: biggest visible frame
        candidates = [fr for fr in page.frames if fr != page.main_frame]
        if not candidates:
            raise RuntimeError("No game frame detected. Open devtools and update frame logic.")
        return candidates[-1]

    def _click_start_buttons(self, page, frame: Frame) -> None:
        for selector in self.config.start_selectors:
            try:
                locator = frame.locator(selector)
                if locator.count() > 0:
                    locator.first.click(timeout=1500)
                    time.sleep(0.5)
                    continue
            except Exception:
                pass
            try:
                locator = page.locator(selector)
                if locator.count() > 0:
                    locator.first.click(timeout=1500)
                    time.sleep(0.5)
            except Exception:
                pass

    def _capture_state(self, frame: Frame) -> Tuple[np.ndarray, List[int]]:
        png = frame.screenshot(type="png")
        img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode frame screenshot")
        board, coming = self.detector.detect(img)
        return board, coming

    def _play_action(self, frame: Frame, action: Tuple[int, int, int, int]) -> None:
        sr, sc, tr, tc = action
        x0, y0 = self.config.board_top_left
        step = self.config.cell_size

        sx = x0 + sc * step + step // 2
        sy = y0 + sr * step + step // 2
        tx = x0 + tc * step + step // 2
        ty = y0 + tr * step + step // 2

        frame.mouse.click(sx, sy)
        time.sleep(self.config.click_delay_s)
        frame.mouse.click(tx, ty)


def main() -> None:
    parser = argparse.ArgumentParser(description="CrazyGames Blooming Garden bot")
    parser.add_argument("--config", default="crazygames_config.json")
    parser.add_argument("--centers", default="centers.txt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    cfg = BotConfig.load(Path(args.config))
    planner_cfg = PlannerConfig(
        beam_width=args.beam_width,
        lookahead_depth=args.depth,
        placement_samples=args.samples,
    )

    bot = CrazyGamesBot(cfg, Path(args.centers), planner_cfg)
    bot.run(steps=args.steps, headless=args.headless)


if __name__ == "__main__":
    main()
