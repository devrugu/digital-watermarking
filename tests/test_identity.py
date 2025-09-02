import cv2
import numpy as np
import watermarking as wm

def _as01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 127).astype(np.uint8)

def test_identity_color(tmp_path, host_color_png, watermark_same_png, key):
    stego = tmp_path / "stego.png"
    extracted = tmp_path / "extracted.png"

    # Strict identity (no fit, no binarize), multi-channel redundancy BGR
    psnr, _ = wm.embed(
        str(host_color_png),
        str(watermark_same_png),
        str(stego),
        wm.EmbedConfig(channels="BGR", key=key, fit=False, binarize=False, progress_step=0),
    )
    assert psnr > 38.0  # 1-LSB embedding should be comfortably high

    wm.extract(
        str(stego),
        str(extracted),
        wm.ExtractConfig(channels="BGR", key=key, progress_step=0),
    )

    ref = cv2.imread(str(watermark_same_png), cv2.IMREAD_UNCHANGED)
    got = cv2.imread(str(extracted), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(_as01(ref), _as01(got)), "extracted should equal input watermark (strict identity)"

def test_identity_gray(tmp_path, host_gray_png, watermark_same_png, key):
    stego = tmp_path / "stego.png"
    extracted = tmp_path / "extracted.png"

    psnr, _ = wm.embed(
        str(host_gray_png),
        str(watermark_same_png),
        str(stego),
        wm.EmbedConfig(channels="B", key=key, fit=False, binarize=False, progress_step=0),
    )
    assert psnr > 38.0

    wm.extract(
        str(stego),
        str(extracted),
        wm.ExtractConfig(channels="B", key=key, progress_step=0),
    )

    ref = cv2.imread(str(watermark_same_png), cv2.IMREAD_UNCHANGED)
    got = cv2.imread(str(extracted), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(_as01(ref), _as01(got))
