"""Take a screenshot of the CrazyGames game frame for manual coordinate calibration.

Use this after layout/site updates to refresh `crazygames_config.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.crazygames.com/game/blooming-garden")
    parser.add_argument("--out", default="results/crazygames_frame.png")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_timeout(7000)

        frame = None
        for fr in page.frames:
            url = (fr.url or "").lower()
            if "crazygames" in url and "game" in url:
                frame = fr
        if frame is None:
            frame = page.main_frame

        frame.screenshot(path=str(out))
        print(f"saved {out}")
        browser.close()


if __name__ == "__main__":
    main()
