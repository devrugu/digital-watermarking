import cv2
import numpy as np
import watermarking as wm

def _as01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 127).astype(np.uint8)

def test_majority_vote_recovers_when_one_channel_corrupted(tmp_path, host_color_png, watermark_same_png, key):
    stego = tmp_path / "stego.png"
    extracted = tmp_path / "extracted.png"

    wm.embed(
        str(host_color_png), str(watermark_same_png), str(stego),
        wm.EmbedConfig(channels="BGR", key=key, fit=False, binarize=False, progress_step=0),
    )

    # Corrupt the LSBs of one channel ('B') in ~30% pixels (random mask)
    img = cv2.imread(str(stego), cv2.IMREAD_UNCHANGED)
    rng = np.random.RandomState(42)
    mask = (rng.rand(img.shape[0], img.shape[1]) < 0.30)
    img[mask, 0] ^= 1  # flip B-channel LSB
    cv2.imwrite(str(stego), img)

    wm.extract(str(stego), str(extracted), wm.ExtractConfig(channels="BGR", key=key, progress_step=0))

    ref = cv2.imread(str(watermark_same_png), cv2.IMREAD_UNCHANGED)
    got = cv2.imread(str(extracted), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(_as01(ref), _as01(got)), "majority vote across B,G,R should overcome one-channel corruption"

def test_tamper_heatmap_and_mask(tmp_path, host_color_png, watermark_same_png, key):
    stego = tmp_path / "stego.png"
    extracted = tmp_path / "extracted.png"
    heatmap = tmp_path / "heatmap.png"
    mask = tmp_path / "mask.png"

    block = 8
    wm.embed(
        str(host_color_png), str(watermark_same_png), str(stego),
        wm.EmbedConfig(channels="B", key=key, fit=False, binarize=False, progress_step=0),
    )

    # Tamper a whole aligned 8x8 block in B channel to trigger detection
    img = cv2.imread(str(stego), cv2.IMREAD_UNCHANGED)
    y0, x0 = 16, 24
    img[y0:y0+block, x0:x0+block, 0] ^= 1  # flip LSBs for the whole block
    cv2.imwrite(str(stego), img)

    wm.extract(
        str(stego),
        str(extracted),
        wm.ExtractConfig(
            channels="B", key=key,
            ref_wm_path=str(watermark_same_png), ref_wm_fit=False, ref_wm_binarize=False,
            tamper_heatmap_path=str(heatmap), tamper_mask_path=str(mask), tamper_threshold=0.10,
            progress_step=0,
        ),
    )

    # Check that the tampered 8x8 region is flagged in mask
    mask_img = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
    assert mask_img is not None
    region = mask_img[y0:y0+block, x0:x0+block]
    assert int(region.mean()) == 255, "tampered block must be fully flagged"

    # Heatmap in that region should be high, and elsewhere mostly low
    heat = cv2.imread(str(heatmap), cv2.IMREAD_UNCHANGED)
    assert heat is not None
    hot = int(heat[y0:y0+block, x0:x0+block].mean())
    cold = int(heat.mean())
    assert hot > 200, "tampered block should show high mismatch"
    assert cold < 80, "overall heatmap average should be relatively low"
