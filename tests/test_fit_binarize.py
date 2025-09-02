import cv2
import numpy as np
import watermarking as wm

def _as01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 127).astype(np.uint8)

def test_fit_and_binarize_roundtrip(tmp_path, host_color_png, watermark_nonbinary_png, key):
    stego = tmp_path / "stego.png"
    extracted = tmp_path / "extracted.png"
    used_bits = tmp_path / "used_wm.png"

    # Flexible mode: resize + binarize, and save the fitted bits
    wm.embed(
        str(host_color_png),
        str(watermark_nonbinary_png),
        str(stego),
        wm.EmbedConfig(channels="BGR", key=key, fit=True, binarize=True, save_fitted_wm=str(used_bits), progress_step=0),
    )

    wm.extract(
        str(stego),
        str(extracted),
        wm.ExtractConfig(channels="BGR", key=key, progress_step=0),
    )

    got = cv2.imread(str(extracted), cv2.IMREAD_UNCHANGED)
    used = cv2.imread(str(used_bits), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(_as01(got), _as01(used)), "extraction must equal the fitted watermark used for embedding"
