import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

# Ensure repo root (containing watermarking.py) is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reproducibility
np.random.seed(1234)

def make_color_host(h, w):
    """Color BGR host with mild gradients and a touch of noise."""
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    b = (x * 0.8 + y * 0.2).astype(np.uint8)
    g = (x * 0.2 + y * 0.8).astype(np.uint8)
    r = ((x + y) // 2).astype(np.uint8)
    img = np.dstack([b, g, r])
    noise = (np.random.randn(h, w, 3) * 2).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def make_gray_host(h, w):
    """Grayscale host."""
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img = ((x * 0.6 + y * 0.4)).astype(np.uint8)
    noise = (np.random.randn(h, w) * 2).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def make_binary_watermark(h, w, block=8):
    """Binary (0/255) watermark with 8x8 checker pattern (block-wise)."""
    yy, xx = np.indices((h, w))
    tiles = ((yy // block + xx // block) % 2).astype(np.uint8) * 255
    return tiles

def make_nonbinary_watermark(h, w):
    """Non-binary grayscale watermark (for --binarize test)."""
    yy, xx = np.indices((h, w))
    img = ((np.sin(xx / 8.0) * 0.5 + 0.5) * 255).astype(np.uint8)
    img = (0.7 * img + 0.3 * yy.astype(np.float32) / max(1, h - 1) * 255).astype(np.uint8)
    return img

@pytest.fixture
def key():
    return "unit-test-key"

@pytest.fixture
def sizes():
    # Use sizes that are *not* multiples of 8 to exercise padding paths
    return dict(h=66, w=70)

@pytest.fixture
def host_color_png(tmp_path, sizes):
    img = make_color_host(sizes["h"], sizes["w"])
    p = tmp_path / "host_color.png"
    cv2.imwrite(str(p), img)
    return p

@pytest.fixture
def host_gray_png(tmp_path, sizes):
    img = make_gray_host(sizes["h"], sizes["w"])
    p = tmp_path / "host_gray.png"
    cv2.imwrite(str(p), img)
    return p

@pytest.fixture
def watermark_same_png(tmp_path, sizes):
    wm = make_binary_watermark(sizes["h"], sizes["w"])
    p = tmp_path / "wm_same.png"
    cv2.imwrite(str(p), wm)
    return p

@pytest.fixture
def watermark_nonbinary_png(tmp_path, sizes):
    # Smaller and non-binary to force --fit host + --binarize path
    wm = make_nonbinary_watermark(sizes["h"] // 2, sizes["w"] // 2)
    p = tmp_path / "wm_nonbinary.png"
    cv2.imwrite(str(p), wm)
    return p
