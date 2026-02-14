"""Take a screenshot of the CrazyGames game frame for manual coordinate calibration.

Use this after layout/site updates to refresh `crazygames_config.json`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Frame, Page, sync_playwright


def frame_clip(page: Page, frame: Frame, timeout_s: float = 15.0) -> dict | None:
    """Best-effort iframe clip resolution.

    Returns None when the iframe bounding box cannot be resolved in time.
    """
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

        # fallback heuristic: any likely game iframe on page
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.crazygames.com/game/blooming-garden")
    parser.add_argument("--out", default="results/crazygames_frame.png")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wait-ms", type=int, default=7000, help="wait before capture so you can click Play")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

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
            # Do not hard-fail; save full page for manual debugging/calibration.
            page.screenshot(path=str(out), full_page=True)
            print(f"saved full-page fallback screenshot {out} (iframe bounding box unresolved)")
        else:
            page.screenshot(path=str(out), clip=clip)
            print(f"saved {out}")
        browser.close()


if __name__ == "__main__":
    main()
