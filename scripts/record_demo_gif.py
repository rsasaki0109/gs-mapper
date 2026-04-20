"""Record the README hero GIF by driving the scene picker through every
bundled splat in the local ``splat.html`` viewer via Playwright, then let
ffmpeg convert the captured WebM to a looping GIF.

Outputs:
    docs/images/demo-sweep/hero.gif
    artifacts/readme-hero/<webm + palette> (kept around for re-encodes)

Requires Playwright (`pip install playwright && playwright install chromium`),
ffmpeg on PATH, and a display for the non-headless chromium run (the
script sets ``DISPLAY=:1`` by default; override via the env var).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OUT_DIR = REPO / "artifacts" / "readme-hero"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    from playwright.sync_api import sync_playwright

    os.environ["DISPLAY"] = ":1"
    port = _free_port()
    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(DOCS),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.8)
        url = f"http://127.0.0.1:{port}/splat.html"
        scenes = [
            "assets/outdoor-demo/outdoor-demo.splat",
            "assets/outdoor-demo/outdoor-demo-dust3r.splat",
            "assets/outdoor-demo/bag6-mast3r.splat",
            "assets/outdoor-demo/mcd-tuhh-day04.splat",
            "assets/outdoor-demo/mcd-tuhh-day04-mast3r.splat",
        ]
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=False,
                args=["--ignore-gpu-blocklist", "--no-sandbox"],
            )
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=str(OUT_DIR),
                record_video_size={"width": 1280, "height": 720},
            )
            page = ctx.new_page()
            page.add_init_script(
                "Object.defineProperty(document, 'fonts', {get: () => ({ready: Promise.resolve(), load: () => Promise.resolve(), check: () => true})});"
            )

            # Initial load: supervised default.
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            # Give the splat parser + sort a chance to converge.
            page.wait_for_timeout(3500)

            for scene in scenes[1:]:
                page.select_option("#sceneSelect", scene)
                page.wait_for_load_state("domcontentloaded", timeout=20000)
                page.wait_for_timeout(3500)

            # Hold on the last scene briefly so the loop ending lands cleanly.
            page.wait_for_timeout(500)

            raw_video = page.video.path() if page.video else None
            ctx.close()
            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=3)
        except subprocess.TimeoutExpired:
            srv.kill()

    if not raw_video:
        print("playwright did not produce a video", file=sys.stderr)
        return 1

    raw_path = Path(raw_video)
    print(f"raw webm: {raw_path} ({raw_path.stat().st_size} bytes)")

    # Downscale + speed up slightly, crop top chrome so the splat fills more,
    # and write a high-quality GIF with a custom palette.
    target_gif = REPO / "docs" / "images" / "demo-sweep" / "hero.gif"
    target_gif.parent.mkdir(parents=True, exist_ok=True)

    palette = OUT_DIR / "palette.png"
    fps = 12
    scale = 720
    vf = f"fps={fps},scale={scale}:-1:flags=lanczos"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-vf",
            vf + ",palettegen=max_colors=192",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-i",
            str(palette),
            "-filter_complex",
            vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            "-loop",
            "0",
            str(target_gif),
        ],
        check=True,
    )

    print(f"wrote {target_gif} ({target_gif.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
