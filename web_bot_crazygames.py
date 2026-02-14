"""Play Blooming Garden on CrazyGames using the local planner with visual debug overlays."""

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
    drag_move_steps: int
    drag_hover_ms: int
    confirm_target_click: bool
    start_after_clicks: int
    preview_action_ms: int
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
            drag_move_steps=int(raw.get("drag_move_steps", 12)),
            drag_hover_ms=int(raw.get("drag_hover_ms", 250)),
            confirm_target_click=bool(raw.get("confirm_target_click", True)),
            start_after_clicks=int(raw.get("start_after_clicks", 3)),
            preview_action_ms=int(raw.get("preview_action_ms", 5000)),
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
        self.manual_click_count = 0

    def run(self, steps: int = 500, headless: bool = False) -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            self._install_click_logger(page)
            page.goto(self.config.url, wait_until="domcontentloaded")
            page.wait_for_timeout(self.config.wait_ms)

            frame = self._resolve_game_frame(page)
            self._click_start_buttons(page, frame)

            print(
                f"ready: steps={steps} beam={self.planner.cfg.beam_width} depth={self.planner.cfg.lookahead_depth} "
                f"samples={self.planner.cfg.placement_samples} start_after_clicks={self.config.start_after_clicks}"
            )

            if self.config.start_after_clicks > 0:
                self._wait_for_manual_clicks(page, self.config.start_after_clicks)

            for step in range(steps):
                clip = self._frame_clip(page, frame)
                if clip is None:
                    frame = self._resolve_game_frame(page)
                    page.wait_for_timeout(300)
                    print(f"step={step} waiting for stable iframe clip")
                    continue

                board, coming = self._capture_state(page, clip)
                env_like = SimpleNamespace(brd=board, coming=coming, score=0)
                action, decision = self._decide_action(env_like)

                self._draw_board_debug(page, clip, action)
                if self.config.preview_action_ms > 0:
                    page.wait_for_timeout(self.config.preview_action_ms)

                self._play_action(page, clip, action)
                self._log_decision(step, board, coming, action, decision)
                time.sleep(self.config.think_delay_s)

            browser.close()

    def _install_click_logger(self, page: Page) -> None:
        def on_click(source, payload):
            self.manual_click_count += 1
            frame_url = ""
            try:
                frame_url = source["frame"].url
            except Exception:
                pass
            print(
                f"manual-click#{self.manual_click_count} x={payload.get('x')} y={payload.get('y')} "
                f"tag={payload.get('tag')} frame={frame_url[:80]}"
            )

        page.expose_binding("__codex_report_click", on_click)
        page.add_init_script(
            """
            () => {
                if (window.__codex_click_hooked) return;
                window.__codex_click_hooked = true;
                document.addEventListener('click', (e) => {
                    const t = e.target;
                    const tag = t && t.tagName ? t.tagName.toLowerCase() : '';
                    if (window.__codex_report_click) {
                        window.__codex_report_click({x:e.clientX,y:e.clientY,tag});
                    }
                }, true);
            }
            """
        )

    def _wait_for_manual_clicks(self, page: Page, needed: int) -> None:
        print(f"waiting for manual clicks before bot starts: need={needed}")
        while self.manual_click_count < needed:
            page.wait_for_timeout(200)
        print(f"manual click threshold reached ({self.manual_click_count}/{needed}), starting bot loop")

    def _decide_action(self, env_like) -> Tuple[Tuple[int, int, int, int], dict]:
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
        top_parts = [f"{act}:{score:.1f}" for score, act in decision.get("top_candidates", [])]
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

    def _draw_board_debug(self, page: Page, clip: dict, action: Tuple[int, int, int, int]) -> None:
        sr, sc, tr, tc = action
        x0, y0 = self.config.board_top_left
        step = self.config.cell_size
        bx, by = clip["x"] + x0, clip["y"] + y0
        board_px = step * 9
        sx, sy = bx + sc * step + step // 2, by + sr * step + step // 2
        tx, ty = bx + tc * step + step // 2, by + tr * step + step // 2

        page.evaluate(
            """([bx,by,boardPx,step,sx,sy,tx,ty,duration]) => {
                const id = 'codex-board-debug';
                const old = document.getElementById(id);
                if (old) old.remove();
                const root = document.createElement('div');
                root.id = id;
                root.style.position = 'fixed';
                root.style.left = '0';
                root.style.top = '0';
                root.style.width = '100vw';
                root.style.height = '100vh';
                root.style.pointerEvents = 'none';
                root.style.zIndex = '2147483646';

                const border = document.createElement('div');
                border.style.position = 'fixed';
                border.style.left = `${bx}px`;
                border.style.top = `${by}px`;
                border.style.width = `${boardPx}px`;
                border.style.height = `${boardPx}px`;
                border.style.border = '3px solid red';
                border.style.boxSizing = 'border-box';
                root.appendChild(border);

                for (let i=1;i<9;i++) {
                    const v = document.createElement('div');
                    v.style.position = 'fixed';
                    v.style.left = `${bx + i*step}px`;
                    v.style.top = `${by}px`;
                    v.style.width = '1px';
                    v.style.height = `${boardPx}px`;
                    v.style.background = 'rgba(255,0,0,0.5)';
                    root.appendChild(v);

                    const h = document.createElement('div');
                    h.style.position = 'fixed';
                    h.style.left = `${bx}px`;
                    h.style.top = `${by + i*step}px`;
                    h.style.width = `${boardPx}px`;
                    h.style.height = '1px';
                    h.style.background = 'rgba(255,0,0,0.5)';
                    root.appendChild(h);
                }

                const mk = (x,y,color,label) => {
                    const c = document.createElement('div');
                    c.style.position='fixed';
                    c.style.left = `${x-14}px`;
                    c.style.top = `${y-14}px`;
                    c.style.width='28px';
                    c.style.height='28px';
                    c.style.border=`4px solid ${color}`;
                    c.style.borderRadius='50%';
                    c.style.boxShadow=`0 0 12px ${color}`;
                    c.style.background='rgba(255,255,255,0.18)';
                    root.appendChild(c);

                    const t = document.createElement('div');
                    t.textContent = label;
                    t.style.position='fixed';
                    t.style.left = `${x+16}px`;
                    t.style.top = `${y-10}px`;
                    t.style.color = color;
                    t.style.font='bold 14px sans-serif';
                    t.style.textShadow='0 0 3px black';
                    root.appendChild(t);
                };

                mk(sx,sy,'red','SRC');
                mk(tx,ty,'yellow','DST');

                document.body.appendChild(root);
                setTimeout(() => { const cur = document.getElementById(id); if (cur) cur.remove(); }, duration);
            }
            """,
            [bx, by, board_px, step, sx, sy, tx, ty, self.config.preview_action_ms],
        )

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

        page.mouse.click(abs_sx, abs_sy)
        time.sleep(self.config.click_delay_s)
        page.mouse.move(abs_sx, abs_sy)
        page.mouse.move(abs_tx, abs_ty, steps=max(1, self.config.drag_move_steps))
        if self.config.drag_hover_ms > 0:
            page.wait_for_timeout(self.config.drag_hover_ms)
        page.mouse.click(abs_tx, abs_ty)
        if self.config.confirm_target_click:
            time.sleep(self.config.click_delay_s)
            page.mouse.click(abs_tx, abs_ty)

        if self.config.show_click_overlay:
            self._visualize_click(page, abs_sx, abs_sy, "#33cc33")
            self._visualize_click(page, abs_tx, abs_ty, "#ff4444")

        print(f"click-flow src=({abs_sx:.1f},{abs_sy:.1f}) -> dst=({abs_tx:.1f},{abs_ty:.1f})")

    def _visualize_click(self, page: Page, x: float, y: float, color: str) -> None:
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
    parser.add_argument("--wait-ms", type=int, default=None, help="initial warmup wait before click-gated start")
    parser.add_argument("--start-after-clicks", type=int, default=None, help="wait for N user clicks before bot starts")
    parser.add_argument("--preview-action-ms", type=int, default=None, help="keep board/action debug overlay before acting")
    parser.add_argument("--hide-click-overlay", action="store_true")
    parser.add_argument("--overlay-duration-ms", type=int, default=None)
    parser.add_argument("--drag-move-steps", type=int, default=None)
    parser.add_argument("--drag-hover-ms", type=int, default=None)
    parser.add_argument("--no-confirm-target-click", action="store_true")
    args = parser.parse_args()

    cfg = BotConfig.load(Path(args.config))
    if args.wait_ms is not None:
        cfg.wait_ms = args.wait_ms
    if args.start_after_clicks is not None:
        cfg.start_after_clicks = args.start_after_clicks
    if args.preview_action_ms is not None:
        cfg.preview_action_ms = args.preview_action_ms
    if args.hide_click_overlay:
        cfg.show_click_overlay = False
    if args.overlay_duration_ms is not None:
        cfg.overlay_duration_ms = args.overlay_duration_ms
    if args.drag_move_steps is not None:
        cfg.drag_move_steps = args.drag_move_steps
    if args.drag_hover_ms is not None:
        cfg.drag_hover_ms = args.drag_hover_ms
    if args.no_confirm_target_click:
        cfg.confirm_target_click = False

    planner_cfg = PlannerConfig(
        beam_width=args.beam_width,
        lookahead_depth=args.depth,
        placement_samples=args.samples,
    )

    bot = CrazyGamesBot(cfg, Path(args.centers), planner_cfg)
    bot.run(steps=args.steps, headless=args.headless)


if __name__ == "__main__":
    main()
