import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np

from .conftest import make_color_host, make_binary_watermark

def _as01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 127).astype(np.uint8)

def test_cli_embed_extract(tmp_path):
    # Create tiny images so this test is fast
    h, w = 40, 48
    host = make_color_host(h, w)
    wm = make_binary_watermark(h, w)

    host_p = tmp_path / "host.png"
    wm_p = tmp_path / "wm.png"
    stego_p = tmp_path / "stego.png"
    ext_p = tmp_path / "extracted.png"

    cv2.imwrite(str(host_p), host)
    cv2.imwrite(str(wm_p), wm)

    # Embed via CLI
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "watermarking.py"),
         "--log-level", "ERROR",
         "embed",
         "--host", str(host_p),
         "--wm", str(wm_p),
         "--out", str(stego_p),
         "--key", "cli-key",
         "--channels", "B"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert stego_p.exists(), "stego.png should be created by CLI"

    # Extract via CLI
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "watermarking.py"),
         "--log-level", "ERROR",
         "extract",
         "--stego", str(stego_p),
         "--out", str(ext_p),
         "--key", "cli-key",
         "--channels", "B"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert ext_p.exists(), "extracted.png should be created by CLI"

    # Verify identity
    ext = cv2.imread(str(ext_p), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(wm_p), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(_as01(ext), _as01(ref)), "CLI roundtrip must preserve watermark exactly"
