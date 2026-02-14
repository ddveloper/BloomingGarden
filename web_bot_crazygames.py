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
from playwright.sync_api import Frame, Page, sync_playwright

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
    wait_ms: int
    show_click_overlay: bool
    overlay_duration_ms: int
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
            wait_ms=int(raw.get("wait_ms", 3000)),
            show_click_overlay=bool(raw.get("show_click_overlay", True)),
            overlay_duration_ms=int(raw.get("overlay_duration_ms", 450)),
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
        return int(np.argmin(dists))


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
            page.wait_for_timeout(self.config.wait_ms)

            frame = self._resolve_game_frame(page)
            self._click_start_buttons(page, frame)

            print(f"ready: wait_ms={self.config.wait_ms} steps={steps} beam={self.planner.cfg.beam_width} depth={self.planner.cfg.lookahead_depth} samples={self.planner.cfg.placement_samples}")

            for step in range(steps):
                clip = self._frame_clip(page, frame)
                if clip is None:
                    # game iframe is often recreated after ads/consent overlays
                    frame = self._resolve_game_frame(page)
                    page.wait_for_timeout(300)
                    print(f"step={step} waiting for stable iframe clip")
                    continue
                board, coming = self._capture_state(page, clip)
                env_like = SimpleNamespace(brd=board, coming=coming, score=0)
                action, decision = self._decide_action(env_like)
                self._play_action(page, clip, action)
                self._log_decision(step, board, coming, action, decision)
                time.sleep(self.config.think_delay_s)

            browser.close()

    def _decide_action(self, env_like) -> Tuple[Tuple[int, int, int, int], dict]:
        """Return selected action plus debugging details for top candidates."""
        actions = self.planner._legal_actions(env_like.brd)
        if not actions:
            return (0, 0, 0, 1), {"top_candidates": [], "action_count": 0}

        scored = sorted(
            ((self.planner._quick_action_score(env_like.brd, action), action) for action in actions),
            reverse=True,
        )
        top = scored[:3]
        action = self.planner.choose_action(env_like)
        return action, {
            "action_count": len(actions),
            "top_candidates": [(float(score), a) for score, a in top],
        }

    def _log_decision(self, step: int, board: np.ndarray, coming: List[int], action: Tuple[int, int, int, int], decision: dict) -> None:
        empty = int(np.count_nonzero(board == EMPTY))
        top_parts = []
        for score, act in decision.get("top_candidates", []):
            top_parts.append(f"{act}:{score:.1f}")
        top_txt = " | ".join(top_parts) if top_parts else "n/a"
        print(
            f"step={step} empty={empty} coming={coming} actions={decision.get('action_count',0)} "
            f"chosen={action} top3={top_txt}"
        )

    def _resolve_game_frame(self, page: Page) -> Frame:
        page.wait_for_timeout(4000)
        for fr in page.frames:
            name = (fr.name or "").lower()
            url = (fr.url or "").lower()
            if "crazygames" in url and ("gameframe" in name or "game" in url):
                return fr
        candidates = [fr for fr in page.frames if fr != page.main_frame]
        if not candidates:
            raise RuntimeError("No game frame detected. Open devtools and update frame logic.")
        return candidates[-1]

    def _frame_clip(self, page: Page, frame: Frame, timeout_s: float = 2.0) -> dict | None:
        if frame == page.main_frame:
            vp = page.viewport_size or {"width": 1600, "height": 1000}
            return {"x": 0, "y": 0, "width": vp["width"], "height": vp["height"]}

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                element = frame.frame_element()
                box = element.bounding_box()
                if box and box["width"] > 0 and box["height"] > 0:
                    return {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
            except Exception:
                pass
            page.wait_for_timeout(100)
        return None

    def _click_start_buttons(self, page: Page, frame: Frame) -> None:
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

    def _capture_state(self, page: Page, clip: dict) -> Tuple[np.ndarray, List[int]]:
        png = page.screenshot(type="png", clip=clip)
        img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode frame screenshot")
        return self.detector.detect(img)

    def _play_action(self, page: Page, clip: dict, action: Tuple[int, int, int, int]) -> None:
        sr, sc, tr, tc = action
        x0, y0 = self.config.board_top_left
        step = self.config.cell_size

        sx = x0 + sc * step + step // 2
        sy = y0 + sr * step + step // 2
        tx = x0 + tc * step + step // 2
        ty = y0 + tr * step + step // 2

        abs_sx, abs_sy = clip["x"] + sx, clip["y"] + sy
        abs_tx, abs_ty = clip["x"] + tx, clip["y"] + ty

        self._visualize_click(page, abs_sx, abs_sy, "#33cc33")
        page.mouse.click(abs_sx, abs_sy)
        time.sleep(self.config.click_delay_s)
        self._visualize_click(page, abs_tx, abs_ty, "#ff4444")
        page.mouse.click(abs_tx, abs_ty)

    def _visualize_click(self, page: Page, x: float, y: float, color: str) -> None:
        if not self.config.show_click_overlay:
            return
        duration = self.config.overlay_duration_ms
        page.evaluate(
            """([x,y,color,duration]) => {
                const dot = document.createElement('div');
                dot.style.position = 'fixed';
                dot.style.left = `${x - 10}px`;
                dot.style.top = `${y - 10}px`;
                dot.style.width = '20px';
                dot.style.height = '20px';
                dot.style.border = `3px solid ${color}`;
                dot.style.borderRadius = '50%';
                dot.style.background = 'rgba(255,255,255,0.15)';
                dot.style.zIndex = '2147483647';
                dot.style.pointerEvents = 'none';
                dot.style.boxShadow = `0 0 10px ${color}`;
                document.body.appendChild(dot);
                setTimeout(() => dot.remove(), duration);
            }
            """,
            [x, y, color, duration],
        )



def main() -> None:
    parser = argparse.ArgumentParser(description="CrazyGames Blooming Garden bot")
    parser.add_argument("--config", default="crazygames_config.json")
    parser.add_argument("--centers", default="centers.txt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--wait-ms", type=int, default=None, help="extra wait before first capture for manual Play click")
    parser.add_argument("--hide-click-overlay", action="store_true", help="disable on-screen click highlight")
    parser.add_argument("--overlay-duration-ms", type=int, default=None, help="highlight duration for each click marker")
    args = parser.parse_args()

    cfg = BotConfig.load(Path(args.config))
    if args.wait_ms is not None:
        cfg.wait_ms = args.wait_ms
    if args.hide_click_overlay:
        cfg.show_click_overlay = False
    if args.overlay_duration_ms is not None:
        cfg.overlay_duration_ms = args.overlay_duration_ms
    planner_cfg = PlannerConfig(
        beam_width=args.beam_width,
        lookahead_depth=args.depth,
        placement_samples=args.samples,
    )

    bot = CrazyGamesBot(cfg, Path(args.centers), planner_cfg)
    bot.run(steps=args.steps, headless=args.headless)


if __name__ == "__main__":
    main()
