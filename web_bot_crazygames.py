"""Play Blooming Garden on CrazyGames using the local planner with perspective-aware debug overlays."""

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
    board_quad: List[Tuple[float, float]]  # lt, lb, rt, rb in frame-local coordinates
    board_top_left: Tuple[int, int]  # legacy fallback
    cell_size: int  # legacy fallback
    next_slots: List[Tuple[int, int, int, int]]
    click_delay_s: float
    think_delay_s: float
    wait_ms: int
    show_click_overlay: bool
    overlay_duration_ms: int
    drag_move_steps: int
    drag_hover_ms: int
    confirm_target_click: bool
    preview_action_ms: int
    board_warp_size: int
    show_cell_labels: bool
    start_selectors: List[str]

    @staticmethod
    def load(path: Path) -> "BotConfig":
        raw = json.loads(path.read_text())
        board_top_left = tuple(raw.get("board_top_left", [220, 130]))
        cell_size = int(raw.get("cell_size", 54))
        board_quad = raw.get("board_quad")
        if not board_quad:
            x0, y0 = board_top_left
            s = cell_size * 9
            board_quad = [[x0, y0], [x0, y0 + s], [x0 + s, y0], [x0 + s, y0 + s]]
        return BotConfig(
            url=raw["url"],
            board_quad=[tuple(x) for x in board_quad],
            board_top_left=board_top_left,
            cell_size=cell_size,
            next_slots=[tuple(x) for x in raw["next_slots"]],
            click_delay_s=float(raw.get("click_delay_s", 0.08)),
            think_delay_s=float(raw.get("think_delay_s", 0.2)),
            wait_ms=int(raw.get("wait_ms", 3000)),
            show_click_overlay=bool(raw.get("show_click_overlay", True)),
            overlay_duration_ms=int(raw.get("overlay_duration_ms", 450)),
            drag_move_steps=int(raw.get("drag_move_steps", 12)),
            drag_hover_ms=int(raw.get("drag_hover_ms", 250)),
            confirm_target_click=bool(raw.get("confirm_target_click", True)),
            preview_action_ms=int(raw.get("preview_action_ms", 5000)),
            board_warp_size=int(raw.get("board_warp_size", 900)),
            show_cell_labels=bool(raw.get("show_cell_labels", True)),
            start_selectors=list(raw.get("start_selectors", [])),
        )


class PerspectiveBoardMapper:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.src = np.array(cfg.board_quad, dtype=np.float32)
        s = float(cfg.board_warp_size - 1)
        self.dst = np.array([(0, 0), (0, s), (s, 0), (s, s)], dtype=np.float32)
        self.h = cv2.getPerspectiveTransform(self.src, self.dst)  # frame -> canonical
        self.h_inv = cv2.getPerspectiveTransform(self.dst, self.src)  # canonical -> frame
        self.cell = cfg.board_warp_size / 9.0

    def warp(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame_bgr, self.h, (self.cfg.board_warp_size, self.cfg.board_warp_size))

    def board_cell_center_frame(self, r: int, c: int) -> Tuple[float, float]:
        pt = np.array([[[((c + 0.5) * self.cell), ((r + 0.5) * self.cell)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.h_inv)[0, 0]
        return float(out[0]), float(out[1])

    def grid_segments_frame(self) -> List[Tuple[float, float, float, float]]:
        segs: List[Tuple[float, float, float, float]] = []
        board_max = float(self.cfg.board_warp_size - 1)
        for i in range(10):
            x = i * self.cell
            p0 = cv2.perspectiveTransform(np.array([[[x, 0.0]]], dtype=np.float32), self.h_inv)[0, 0]
            p1 = cv2.perspectiveTransform(np.array([[[x, board_max]]], dtype=np.float32), self.h_inv)[0, 0]
            segs.append((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])))

            y = i * self.cell
            q0 = cv2.perspectiveTransform(np.array([[[0.0, y]]], dtype=np.float32), self.h_inv)[0, 0]
            q1 = cv2.perspectiveTransform(np.array([[[board_max, y]]], dtype=np.float32), self.h_inv)[0, 0]
            segs.append((float(q0[0]), float(q0[1]), float(q1[0]), float(q1[1])))
        return segs

    def border_points_frame(self) -> List[Tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.src.tolist()]


class CrazyGamesBoardDetector:
    def __init__(self, cfg: BotConfig, centers_file: Path):
        self.cfg = cfg
        self.mapper = PerspectiveBoardMapper(cfg)
        self.centers = np.loadtxt(centers_file, dtype=np.float32)
        if self.centers.shape != (8, 2):
            raise ValueError("centers.txt should contain 8x2 cluster centers")

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        board = np.full((9, 9), EMPTY, dtype=int)

        warped = self.mapper.warp(frame_bgr)
        ycc_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2YCR_CB)
        cell = self.mapper.cell
        patch_r = max(3, int(cell * 0.08))

        for r in range(9):
            for c in range(9):
                cx = int((c + 0.5) * cell)
                cy = int((r + 0.5) * cell)
                patch = ycc_warp[max(0, cy - patch_r) : cy + patch_r, max(0, cx - patch_r) : cx + patch_r, 1:]
                feat = np.mean(patch, axis=(0, 1))
                board[r][c] = self._nearest_flower(feat)

        next_flowers: List[int] = []
        ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCR_CB)
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

            print(
                f"ready: steps={steps} beam={self.planner.cfg.beam_width} depth={self.planner.cfg.lookahead_depth} "
                f"samples={self.planner.cfg.placement_samples}"
            )

            self._wait_for_user_confirmation()

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
                self._log_detected_board(board, coming)

                self._draw_board_debug(page, clip, action, board)
                if self.config.preview_action_ms > 0:
                    page.wait_for_timeout(self.config.preview_action_ms)

                self._play_action(page, clip, action)
                self._log_decision(step, board, coming, action, decision)
                time.sleep(self.config.think_delay_s)

            browser.close()

    def _wait_for_user_confirmation(self) -> None:
        while True:
            ans = input("Type y then Enter to start bot actions: ").strip().lower()
            if ans == "y":
                print("received y, starting bot loop")
                return
            print("not started. please type y to start.")

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
        return action, {"action_count": len(actions), "top_candidates": [(float(score), a) for score, a in top]}

    def _log_detected_board(self, board: np.ndarray, coming: List[int]) -> None:
        flowers = []
        for r in range(9):
            for c in range(9):
                val = int(board[r][c])
                if val != EMPTY:
                    flowers.append((r, c, val))
        print(f"detected-coming={coming} flower-count={len(flowers)} flowers={flowers[:40]}{' ...' if len(flowers)>40 else ''}")

    def _log_decision(self, step: int, board: np.ndarray, coming: List[int], action: Tuple[int, int, int, int], decision: dict) -> None:
        empty = int(np.count_nonzero(board == EMPTY))
        top_parts = [f"{act}:{score:.1f}" for score, act in decision.get("top_candidates", [])]
        top_txt = " | ".join(top_parts) if top_parts else "n/a"
        print(f"step={step} empty={empty} coming={coming} actions={decision.get('action_count',0)} chosen={action} top3={top_txt}")

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
                box = frame.frame_element().bounding_box()
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

    def _draw_board_debug(self, page: Page, clip: dict, action: Tuple[int, int, int, int], board: np.ndarray) -> None:
        sr, sc, tr, tc = action
        mapper = self.detector.mapper
        border = [(clip["x"] + x, clip["y"] + y) for x, y in mapper.border_points_frame()]
        segs = [
            (clip["x"] + x1, clip["y"] + y1, clip["x"] + x2, clip["y"] + y2)
            for x1, y1, x2, y2 in mapper.grid_segments_frame()
        ]
        sx, sy = mapper.board_cell_center_frame(sr, sc)
        tx, ty = mapper.board_cell_center_frame(tr, tc)
        sx += clip["x"]; sy += clip["y"]
        tx += clip["x"]; ty += clip["y"]

        labels = []
        if self.config.show_cell_labels:
            for r in range(9):
                for c in range(9):
                    val = int(board[r][c])
                    if val == EMPTY:
                        continue
                    lx, ly = mapper.board_cell_center_frame(r, c)
                    labels.append((clip["x"] + lx, clip["y"] + ly, str(val)))

        page.evaluate(
            """([border,segs,sx,sy,tx,ty,labels,duration]) => {
                const id = 'codex-board-debug';
                const old = document.getElementById(id);
                if (old) old.remove();

                const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
                svg.id = id;
                svg.setAttribute('width', window.innerWidth);
                svg.setAttribute('height', window.innerHeight);
                svg.style.position = 'fixed';
                svg.style.left = '0';
                svg.style.top = '0';
                svg.style.pointerEvents = 'none';
                svg.style.zIndex = '2147483646';

                const poly = document.createElementNS('http://www.w3.org/2000/svg','polygon');
                poly.setAttribute('points', border.map(p => `${p[0]},${p[1]}`).join(' '));
                poly.setAttribute('fill', 'none');
                poly.setAttribute('stroke', 'red');
                poly.setAttribute('stroke-width', '3');
                svg.appendChild(poly);

                for (const s of segs) {
                    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
                    line.setAttribute('x1', s[0]); line.setAttribute('y1', s[1]);
                    line.setAttribute('x2', s[2]); line.setAttribute('y2', s[3]);
                    line.setAttribute('stroke', 'rgba(255,0,0,0.55)');
                    line.setAttribute('stroke-width', '1');
                    svg.appendChild(line);
                }

                const mk = (x,y,color,label) => {
                    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
                    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 14);
                    c.setAttribute('fill', 'rgba(255,255,255,0.18)');
                    c.setAttribute('stroke', color); c.setAttribute('stroke-width', 4);
                    svg.appendChild(c);

                    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
                    t.setAttribute('x', x + 18); t.setAttribute('y', y + 4);
                    t.setAttribute('fill', color); t.setAttribute('font-size', 14); t.setAttribute('font-weight', 'bold');
                    t.textContent = label;
                    svg.appendChild(t);
                };
                mk(sx,sy,'red','SRC');
                mk(tx,ty,'yellow','DST');

                for (const lab of labels) {
                    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
                    t.setAttribute('x', lab[0] + 6);
                    t.setAttribute('y', lab[1] - 6);
                    t.setAttribute('fill', '#00ffff');
                    t.setAttribute('font-size', 12);
                    t.setAttribute('font-weight', 'bold');
                    t.textContent = lab[2];
                    svg.appendChild(t);
                }

                document.body.appendChild(svg);
                setTimeout(() => { const cur = document.getElementById(id); if (cur) cur.remove(); }, duration);
            }
            """,
            [border, segs, sx, sy, tx, ty, labels, self.config.preview_action_ms],
        )

    def _play_action(self, page: Page, clip: dict, action: Tuple[int, int, int, int]) -> None:
        sr, sc, tr, tc = action
        mapper = self.detector.mapper
        sx, sy = mapper.board_cell_center_frame(sr, sc)
        tx, ty = mapper.board_cell_center_frame(tr, tc)
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
    parser.add_argument("--wait-ms", type=int, default=None, help="initial warmup wait before CLI-start prompt")
    parser.add_argument("--preview-action-ms", type=int, default=None, help="keep board/action debug overlay before acting")
    parser.add_argument("--hide-click-overlay", action="store_true")
    parser.add_argument("--overlay-duration-ms", type=int, default=None)
    parser.add_argument("--drag-move-steps", type=int, default=None)
    parser.add_argument("--drag-hover-ms", type=int, default=None)
    parser.add_argument("--no-confirm-target-click", action="store_true")
    parser.add_argument("--hide-cell-labels", action="store_true", help="hide detected flower labels on overlay")
    args = parser.parse_args()

    cfg = BotConfig.load(Path(args.config))
    if args.wait_ms is not None:
        cfg.wait_ms = args.wait_ms
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
    if args.hide_cell_labels:
        cfg.show_cell_labels = False

    planner_cfg = PlannerConfig(
        beam_width=args.beam_width,
        lookahead_depth=args.depth,
        placement_samples=args.samples,
    )

    bot = CrazyGamesBot(cfg, Path(args.centers), planner_cfg)
    bot.run(steps=args.steps, headless=args.headless)


if __name__ == "__main__":
    main()
