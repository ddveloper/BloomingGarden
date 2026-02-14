"""Tools for CrazyGames calibration.

Modes:
1) screenshot mode (default): save game frame screenshot.
2) interactive quad tuning: adjust board_quad from CLI and preview red cross markers in browser.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import Frame, Page, sync_playwright


def frame_clip(page: Page, frame: Frame, timeout_s: float = 15.0) -> dict | None:
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
        try:
            loc = page.locator("iframe[src*='crazygames'], iframe[src*='game'], iframe")
            if loc.count() > 0:
                box = loc.first.bounding_box()
                if box and box["width"] > 0 and box["height"] > 0:
                    return {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
        except Exception:
            pass
        page.wait_for_timeout(250)
    return None


def load_quad(config_path: Path) -> list[list[float]]:
    if not config_path.exists():
        return [[210, 180], [90, 665], [930, 180], [1085, 690]]
    data = json.loads(config_path.read_text())
    quad = data.get("board_quad")
    if quad and len(quad) == 4:
        return [[float(x), float(y)] for x, y in quad]
    return [[210, 180], [90, 665], [930, 180], [1085, 690]]


def save_quad(config_path: Path, quad: list[list[float]]) -> None:
    data = {}
    if config_path.exists():
        data = json.loads(config_path.read_text())
    data["board_quad"] = [[round(x, 1), round(y, 1)] for x, y in quad]
    config_path.write_text(json.dumps(data, indent=2))


def draw_cross_preview(page: Page, clip: dict, quad: list[list[float]], selected: int, hold_ms: int = 2000) -> None:
    abs_quad = [[clip["x"] + p[0], clip["y"] + p[1]] for p in quad]
    page.evaluate(
        """([quad, selected, holdMs]) => {
            const id='codex-quad-preview';
            const old=document.getElementById(id);
            if (old) old.remove();

            const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
            svg.id=id;
            svg.setAttribute('width',window.innerWidth);
            svg.setAttribute('height',window.innerHeight);
            svg.style.position='fixed';
            svg.style.left='0';
            svg.style.top='0';
            svg.style.pointerEvents='none';
            svg.style.zIndex='2147483647';

            const labels=['LT','LB','RT','RB'];
            const poly=document.createElementNS('http://www.w3.org/2000/svg','polygon');
            poly.setAttribute('points', quad.map(p=>`${p[0]},${p[1]}`).join(' '));
            poly.setAttribute('fill','none');
            poly.setAttribute('stroke','rgba(255,0,0,0.8)');
            poly.setAttribute('stroke-width','2');
            svg.appendChild(poly);

            for (let i=0;i<quad.length;i++) {
                const x=quad[i][0], y=quad[i][1];
                const color=i===selected ? 'yellow' : 'red';
                const lw=i===selected ? 4 : 3;

                const l1=document.createElementNS('http://www.w3.org/2000/svg','line');
                l1.setAttribute('x1',x-12); l1.setAttribute('y1',y);
                l1.setAttribute('x2',x+12); l1.setAttribute('y2',y);
                l1.setAttribute('stroke',color); l1.setAttribute('stroke-width',lw);
                svg.appendChild(l1);

                const l2=document.createElementNS('http://www.w3.org/2000/svg','line');
                l2.setAttribute('x1',x); l2.setAttribute('y1',y-12);
                l2.setAttribute('x2',x); l2.setAttribute('y2',y+12);
                l2.setAttribute('stroke',color); l2.setAttribute('stroke-width',lw);
                svg.appendChild(l2);

                const t=document.createElementNS('http://www.w3.org/2000/svg','text');
                t.setAttribute('x',x+14); t.setAttribute('y',y-10);
                t.setAttribute('fill',color); t.setAttribute('font-size',14); t.setAttribute('font-weight','bold');
                t.textContent=labels[i];
                svg.appendChild(t);
            }

            document.body.appendChild(svg);
            setTimeout(()=>{const cur=document.getElementById(id); if (cur) cur.remove();}, holdMs);
        }
        """,
        [abs_quad, selected, hold_ms],
    )


def interactive_quad_tune(page: Page, clip: dict, config_path: Path, step_px: int) -> None:
    quad = load_quad(config_path)
    selected = 0
    step = step_px
    names = ["lt", "lb", "rt", "rb"]

    print("Interactive quad tuning started.")
    print("Commands: 1/2/3/4 or lt/lb/rt/rb select corner | w/a/s/d move | +/- step | show | save | q")

    while True:
        print(f"selected={names[selected]} step={step} quad={[[round(x,1),round(y,1)] for x,y in quad]}")
        cmd = input("quad> ").strip().lower()

        if cmd in {"q", "quit", "exit"}:
            print("exit tuning (no auto-save)")
            return
        if cmd in {"save", "s!"}:
            save_quad(config_path, quad)
            print(f"saved board_quad to {config_path}")
            continue
        if cmd in {"show", "p", ""}:
            draw_cross_preview(page, clip, quad, selected, hold_ms=2000)
            page.wait_for_timeout(2000)
            continue

        if cmd in {"1", "lt"}:
            selected = 0
        elif cmd in {"2", "lb"}:
            selected = 1
        elif cmd in {"3", "rt"}:
            selected = 2
        elif cmd in {"4", "rb"}:
            selected = 3
        elif cmd == "+":
            step += 1
        elif cmd == "-":
            step = max(1, step - 1)
        elif cmd in {"a", "left"}:
            quad[selected][0] -= step
        elif cmd in {"d", "right"}:
            quad[selected][0] += step
        elif cmd in {"w", "up"}:
            quad[selected][1] -= step
        elif cmd in {"s", "down"}:
            quad[selected][1] += step
        else:
            print("unknown command")
            continue

        draw_cross_preview(page, clip, quad, selected, hold_ms=2000)
        page.wait_for_timeout(2000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.crazygames.com/game/blooming-garden")
    parser.add_argument("--out", default="results/crazygames_frame.png")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wait-ms", type=int, default=7000, help="wait before capture so you can click Play")
    parser.add_argument("--interactive-quad", action="store_true", help="interactive board_quad tuning from CLI")
    parser.add_argument("--config", default="crazygames_config.json")
    parser.add_argument("--step-px", type=int, default=5, help="movement step for quad tuning")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_timeout(args.wait_ms)

        frame = None
        for fr in page.frames:
            url = (fr.url or "").lower()
            if "crazygames" in url and "game" in url:
                frame = fr
        if frame is None:
            frame = page.main_frame

        clip = frame_clip(page, frame)
        if clip is None:
            page.screenshot(path=str(out), full_page=True)
            print(f"saved full-page fallback screenshot {out} (iframe bounding box unresolved)")
            browser.close()
            return

        if args.interactive_quad:
            if args.headless:
                raise RuntimeError("interactive quad tuning requires non-headless mode")
            interactive_quad_tune(page, clip, config_path, args.step_px)
        else:
            page.screenshot(path=str(out), clip=clip)
            print(f"saved {out}")

        browser.close()


if __name__ == "__main__":
    main()
