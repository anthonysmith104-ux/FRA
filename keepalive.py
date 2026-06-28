"""keepalive.py — keep Streamlit Community Cloud apps awake.

Streamlit Cloud sleeps an app after ~12 hours without real traffic. A plain
HTTP ping doesn't help — it only fetches the static HTML shell, so the Python
app never starts. This script uses a real headless browser (Playwright) to
actually load each app, run its JavaScript, and click the "wake this app back
up" button if it's found sleeping. Run on a schedule (see the GitHub Actions
workflow) every few hours and the apps stay awake.

Fill in YOUR app URLs below.
"""
from playwright.sync_api import sync_playwright

# ── Your apps — replace/confirm these URLs ──────────────────────────────────
URLS = [
    "https://foresightpi.streamlit.app/",          # advisor app (Proposal Desk)
    "https://riskcheckup.streamlit.app/",          # client portal (Risk Checkup)
]

WAKE_PHRASES = ["get this app back up", "Yes, get this app"]


def visit(page, url):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    except Exception as e:
        print(f"ERR  {url}  ({e})")
        return
    # Give the page a moment to render the (possible) sleep screen.
    page.wait_for_timeout(4_000)
    woke = False
    for phrase in WAKE_PHRASES:
        try:
            btn = page.get_by_text(phrase, exact=False)
            if btn.count() > 0:
                btn.first.click()
                woke = True
                # Wait for the app to actually boot back up.
                page.wait_for_timeout(20_000)
                break
        except Exception:
            pass
    print(f"{'WOKE' if woke else 'OK  '} {url}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for u in URLS:
            if "REPLACE-WITH" in u:
                print(f"SKIP {u}  (placeholder — edit keepalive.py)")
                continue
            visit(page, u)
        browser.close()


if __name__ == "__main__":
    main()
