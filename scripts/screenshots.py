"""Take README screenshots of a running harness (optional; needs Playwright).

    pip install playwright && playwright install chromium
    python -m harness serve --no-browser          # in another terminal
    python scripts/screenshots.py --provider gemini --notebook samples/cash_application_engine.py

Saves docs/screenshots/01-overview.png ... 05-report.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--notebook", default="samples/cash_application_engine.py",
                    help="path as the server sees it (relative to where it was started)")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--cite", default="C29:L4", help="citation to click for the highlight screenshot")
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 960}, device_scale_factor=1.5)
        page.goto(args.url)
        page.fill("#path", args.notebook)
        page.click("#load-form button[type=submit]")
        page.wait_for_selector(".cell")
        page.screenshot(path=str(out / "01-overview.png"))
        page.select_option("#provider", args.provider)
        page.click("#generate")
        # selector waits only: the app's Content-Security-Policy blocks eval-based waits
        page.wait_for_selector("#tabs button[data-tab=agent] .count", timeout=args.timeout * 1000)
        page.click("#tabs button[data-tab=manager]")
        page.wait_for_selector(".claim")
        page.screenshot(path=str(out / "02-manager.png"))
        chip = page.locator(f".chip:text-is('{args.cite}')").first
        if chip.count():
            chip.click()
            page.wait_for_timeout(900)
        page.screenshot(path=str(out / "03-citation-click.png"))
        page.click("#tabs button[data-tab=agent]")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "04-agent.png"))
        page.click("#tabs button[data-tab=report]")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "05-report.png"))
        browser.close()
    print(f"saved screenshots in {out}")


if __name__ == "__main__":
    main()
