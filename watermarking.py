#!/usr/bin/env python3
"""
watermarking.py
===============
Block-based robust image watermarking (PNG in/out), color-friendly,
inspired by Ping Wah Wong's verification scheme.

Key defaults & modes
--------------------
- **Strict identity by default:** no resizing, no thresholding. If your
  `watermark.png` has the SAME size as `host.png` and is already BINARY
  (0 or 255), then `extracted.png` will be pixel-identical to `watermark.png`.
- **Flexible mode:** opt into `--fit host` (resize) and/or `--binarize` (Otsu).
  You can also save the exact watermark used at embed time via `--save-fitted-wm`.

New features
------------
- **Multi-channel embedding/extraction** across any subset of {B,G,R}.
  By default, we embed the SAME watermark into each selected channel,
  which gives redundancy. At extraction, you can combine channels with
  **majority vote** for robustness.
- **Tamper heatmap** (and optional binary tamper mask) when the original
  watermark is provided to extraction; highlights blocks whose extracted
  watermark bits disagree with the original.

Algorithm (practical variant of Wong)
-------------------------------------
Per 8×8 block on each selected channel:
  1) Zero LSBs to form X̃_r.
  2) Compute P_r = HMAC-SHA256(X̃_r || coords || size, key)[0:64 bits].
  3) Form W_r = B_r XOR P_r, where B_r are 64 watermark bits.
  4) Write W_r into the block's LSBs.
Extraction reverses these steps to recover B_r exactly.

CLI quickstart (strict identity)
-------------------------------
Embed:
    python watermarking.py embed \
      --host host.png \
      --wm watermark.png \
      --out stego.png \
      --key "demo" \
      --channels B

Extract:
    python watermarking.py extract \
      --stego stego.png \
      --out extracted.png \
      --key "demo" \
      --channels B

Flexible example (resize + binarize + save fitted watermark):
    python watermarking.py embed \
      --host host.png \
      --wm watermark.png \
      --out stego.png \
      --key "demo" \
      --channels BGR \
      --fit host \
      --binarize \
      --save-fitted-wm used_wm.png

Tamper heatmap (compare with original watermark):
    python watermarking.py extract \
      --stego stego.png \
      --out extracted.png \
      --key "demo" \
      --channels BGR \
      --wm watermark.png \
      --wm-fit host \
      --wm-binarize \
      --tamper-heatmap heatmap.png \
      --tamper-mask tamper_mask.png \
      --tamper-threshold 0.125
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from dataclasses import dataclass
from typing import Tuple, Optional, List, Literal

import cv2
import numpy as np


# --------------------------- Utility Functions --------------------------- #

def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _split_color_and_alpha(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (color_or_gray, alpha_or_None)."""
    if img.ndim == 2:
        return img, None
    if img.shape[2] == 3:
        return img, None
    if img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    raise ValueError("Unsupported channel count")


def _merge_color_and_alpha(color: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    color = _ensure_uint8(color)
    if alpha is None:
        return color
    alpha = _ensure_uint8(alpha)
    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    return np.dstack([color, alpha])


def _pad_to_block(img: np.ndarray, block: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad to multiples of `block` using edge replication. Return (padded, (orig_h,orig_w))."""
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    pad_h = (block - (h % block)) % block
    pad_w = (block - (w % block)) % block
    if pad_h == 0 and pad_w == 0:
        return img, (h, w)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return padded, (h, w)


def _crop_to_shape(img: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return img[:h, :w]


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _binary_from_png(wm: np.ndarray, target_hw: Tuple[int, int], fit: bool, binarize: bool) -> np.ndarray:
    """
    Convert PNG to 0/1 watermark bits. No resize unless fit=True. Otsu threshold only if binarize=True.
    """
    wm_gray = _to_gray(wm)
    H, W = target_hw
    if fit and wm_gray.shape != (H, W):
        wm_gray = cv2.resize(wm_gray, (W, H), interpolation=cv2.INTER_AREA)
    if binarize:
        _, wm_gray = cv2.threshold(wm_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = (wm_gray > 127).astype(np.uint8)
    return bits


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


# ------------------------- Core Watermarking Logic ----------------------- #

def _pr_bits(block_zeroed: np.ndarray, block_xy: Tuple[int, int],
             full_hw: Tuple[int, int], key: bytes) -> np.ndarray:
    """
    64-bit pseudo-random mask via HMAC-SHA256 for deterministic keyed masking.
    Input:
      - block_zeroed: 8x8 uint8 block with LSBs zeroed
      - block_xy: (block_x, block_y)
      - full_hw: block-padded plane height/width
    """
    h = hmac.new(key, digestmod=hashlib.sha256)
    h.update(block_zeroed.tobytes())
    h.update(int(block_xy[0]).to_bytes(4, 'big'))
    h.update(int(block_xy[1]).to_bytes(4, 'big'))
    h.update(int(full_hw[0]).to_bytes(4, 'big'))
    h.update(int(full_hw[1]).to_bytes(4, 'big'))
    digest = h.digest()
    first_8 = np.frombuffer(digest[:8], dtype=np.uint8)  # 64 bits
    return np.unpackbits(first_8).astype(np.uint8)


@dataclass
class EmbedConfig:
    block: int = 8
    channels: str = "B"     # any subset like "B", "G", "R", "BG", "BGR", or "B,G"
    key: str = ""
    fit: bool = False       # strict identity unless True
    binarize: bool = False  # strict identity unless True
    save_fitted_wm: Optional[str] = None  # path to save the exact bits used (optional)


@dataclass
class ExtractConfig:
    block: int = 8
    channels: str = "B"     # same parsing rules as embedding
    key: str = ""
    # tamper reporting (optional)
    ref_wm_path: Optional[str] = None
    ref_wm_fit: bool = False
    ref_wm_binarize: bool = False
    tamper_heatmap_path: Optional[str] = None
    tamper_mask_path: Optional[str] = None
    tamper_threshold: float = 0.0  # fraction in [0,1]; if >0, write binary mask


def _parse_channels(ch_str: str, color_ndim: int) -> List[int]:
    """
    Parse channel selection string into BGR indices.
    Accepts: 'B', 'G', 'R', 'BG', 'BGR', 'B,G', case-insensitive.
    If grayscale image, returns [None] sentinel meaning single plane.
    """
    if color_ndim == 2:
        return [-1]  # sentinel for grayscale
    s = ch_str.replace(",", "").upper()
    valid = {"B": 0, "G": 1, "R": 2}
    seen = []
    for c in s:
        if c not in valid:
            raise ValueError("channels must be a combination of 'B','G','R'")
        idx = valid[c]
        if idx not in seen:
            seen.append(idx)
    if not seen:
        raise ValueError("No valid channels parsed.")
    return seen


def _prepare_wm_bits(host_hw: Tuple[int, int], wm_png_path: str, fit: bool, binarize: bool) -> np.ndarray:
    wm_src = cv2.imread(wm_png_path, cv2.IMREAD_UNCHANGED)
    if wm_src is None:
        raise FileNotFoundError(f"Watermark image not found: {wm_png_path}")
    H, W = host_hw
    wm_bits = _binary_from_png(wm_src, (H, W), fit=fit, binarize=binarize)
    if not fit:
        h0, w0 = _to_gray(wm_src).shape[:2]
        if (h0, w0) != (H, W):
            raise ValueError(
                f"Strict identity requires watermark size == host size ({H}x{W}); got {h0}x{w0}. "
                f"Use --fit host to resize."
            )
    return wm_bits


def embed(host_png_path: str, wm_png_path: str, out_png_path: str,
          cfg: EmbedConfig = EmbedConfig()) -> Tuple[float, str]:
    """
    Embed a watermark into the host image across selected channels.
    Returns (psnr_value, out_path).
    """
    host = cv2.imread(host_png_path, cv2.IMREAD_UNCHANGED)
    if host is None:
        raise FileNotFoundError(f"Host image not found: {host_png_path}")
    host = _ensure_uint8(host)
    color, alpha = _split_color_and_alpha(host)

    # Select channels
    ch_idxs = _parse_channels(cfg.channels, color.ndim)

    # Set up planes list (handles grayscale sentinel -1)
    if ch_idxs == [-1]:
        planes = [color.copy()]
    else:
        planes = [color[:, :, ci].copy() for ci in ch_idxs]

    H, W = planes[0].shape[:2]
    wm_bits = _prepare_wm_bits((H, W), wm_png_path, fit=cfg.fit, binarize=cfg.binarize)

    # Optionally save the exact watermark used
    if cfg.save_fitted_wm:
        cv2.imwrite(cfg.save_fitted_wm, (wm_bits * 255).astype(np.uint8))

    key_bytes = cfg.key.encode("utf-8")

    # Embed SAME bits into each selected plane (redundancy)
    embedded_planes = []
    for plane in planes:
        plane_padded, orig_hw = _pad_to_block(plane, cfg.block)
        wm_padded, _ = _pad_to_block(wm_bits, cfg.block)
        h_pad, w_pad = plane_padded.shape

        for by in range(0, h_pad, cfg.block):
            for bx in range(0, w_pad, cfg.block):
                blk = plane_padded[by:by + cfg.block, bx:bx + cfg.block].copy()
                blk_zeroed = (blk & 0xFE).astype(np.uint8)
                wm_blk = wm_padded[by:by + cfg.block, bx:bx + cfg.block].astype(np.uint8).flatten()
                pr = _pr_bits(blk_zeroed, (bx // cfg.block, by // cfg.block), (h_pad, w_pad), key_bytes)
                wr = (wm_blk ^ pr).reshape(cfg.block, cfg.block)
                blk_zeroed[wr == 1] |= 0x01
                plane_padded[by:by + cfg.block, bx:bx + cfg.block] = blk_zeroed

        embedded_planes.append(_crop_to_shape(plane_padded, orig_hw))

    # Reassemble image
    if color.ndim == 2:
        stego = embedded_planes[0]
    else:
        stego = color.copy()
        for ci, plane_out in zip(ch_idxs, embedded_planes):
            stego[:, :, ci] = plane_out

    out_img = _merge_color_and_alpha(stego, alpha)
    if not cv2.imwrite(out_png_path, out_img):
        raise IOError(f"Failed to write output PNG: {out_png_path}")
    return psnr(host, out_img), out_png_path


def _extract_bits_from_plane(plane: np.ndarray, block: int, key_bytes: bytes) -> np.ndarray:
    plane_padded, orig_hw = _pad_to_block(plane, block)
    h_pad, w_pad = plane_padded.shape
    wm_rec = np.zeros_like(plane_padded, dtype=np.uint8)
    for by in range(0, h_pad, block):
        for bx in range(0, w_pad, block):
            blk = plane_padded[by:by + block, bx:bx + block].copy()
            lsbs = (blk & 0x01).flatten().astype(np.uint8)
            blk_zeroed = (blk & 0xFE).astype(np.uint8)
            pr = _pr_bits(blk_zeroed, (bx // block, by // block), (h_pad, w_pad), key_bytes)
            wm_blk = (pr ^ lsbs).reshape(block, block)  # 0/1
            wm_rec[by:by + block, bx:bx + block] = wm_blk
    return _crop_to_shape(wm_rec, orig_hw)  # 0/1


def _majority_vote(bitmaps: List[np.ndarray]) -> np.ndarray:
    """Majority vote across K bitmaps of shape HxW (values 0/1)."""
    stack = np.stack(bitmaps, axis=0).astype(np.int16)  # KxHxW
    votes = np.sum(stack, axis=0)
    return (votes >= (stack.shape[0] + 1) // 2).astype(np.uint8)


def extract(stego_png_path: str, out_wm_png_path: str, cfg: ExtractConfig = ExtractConfig()) -> str:
    """
    Extract watermark from stego image. If multiple channels are selected,
    majority vote is applied to combine them (unless you only select one).
    If `ref_wm_path` is provided, a tamper heatmap/mask can be written.
    """
    stego = cv2.imread(stego_png_path, cv2.IMREAD_UNCHANGED)
    if stego is None:
        raise FileNotFoundError(f"Stego image not found: {stego_png_path}")
    stego = _ensure_uint8(stego)
    color, _ = _split_color_and_alpha(stego)

    ch_idxs = _parse_channels(cfg.channels, color.ndim)

    # Build per-channel planes list
    if ch_idxs == [-1]:
        planes = [color.copy()]
        labels = ["GRAY"]
    else:
        planes = [color[:, :, ci].copy() for ci in ch_idxs]
        inv = {0: "B", 1: "G", 2: "R"}
        labels = [inv[ci] for ci in ch_idxs]

    key_bytes = cfg.key.encode("utf-8")

    # Extract per channel
    per_channel_bits = [ _extract_bits_from_plane(p, cfg.block, key_bytes) for p in planes ]

    # Combine (vote) if multiple
    if len(per_channel_bits) == 1:
        final_bits = per_channel_bits[0]
    else:
        final_bits = _majority_vote(per_channel_bits)

    # Save final extraction
    if not cv2.imwrite(out_wm_png_path, (final_bits * 255).astype(np.uint8)):
        raise IOError(f"Failed to write output PNG: {out_wm_png_path}")

    # Tamper heatmap/mask if reference watermark is provided
    if cfg.ref_wm_path:
        H, W = final_bits.shape[:2]
        ref_bits = _prepare_wm_bits((H, W), cfg.ref_wm_path, fit=cfg.ref_wm_fit, binarize=cfg.ref_wm_binarize)

        # Per-block mismatch ratio
        block = cfg.block
        heat = np.zeros_like(final_bits, dtype=np.uint8)
        for by in range(0, H, block):
            for bx in range(0, W, block):
                fblk = final_bits[by:by+block, bx:bx+block]
                rblk = ref_bits[by:by+block, bx:bx+block]
                # guard against edge blocks if H,W are not multiples (shouldn't happen due to identity/fitting)
                if fblk.shape != (block, block) or rblk.shape != (block, block):
                    sz = min(fblk.shape[0], rblk.shape[0], block), min(fblk.shape[1], rblk.shape[1], block)
                    fblk = fblk[:sz[0], :sz[1]]
                    rblk = rblk[:sz[0], :sz[1]]
                    denom = float(sz[0] * sz[1])
                else:
                    denom = float(block * block)
                mism = np.sum(fblk != rblk) / denom
                val = int(round(255.0 * mism))
                heat[by:by+fblk.shape[0], bx:bx+fblk.shape[1]] = val

        if cfg.tamper_heatmap_path:
            cv2.imwrite(cfg.tamper_heatmap_path, heat)

        if cfg.tamper_mask_path and cfg.tamper_threshold > 0.0:
            mask = (heat >= int(round(255.0 * cfg.tamper_threshold))).astype(np.uint8) * 255
            cv2.imwrite(cfg.tamper_mask_path, mask)

    return out_wm_png_path


# ------------------------------ CLI ------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Block-LSB watermarking (PNG in/out), color-capable, identity-safe.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # EMBED
    pe = sub.add_parser("embed", help="Embed a watermark into a host PNG.")
    pe.add_argument("--host", required=True, help="Path to host PNG.")
    pe.add_argument("--wm", required=True, help="Path to watermark PNG.")
    pe.add_argument("--out", required=True, help="Output stego PNG path.")
    pe.add_argument("--channels", default="B",
                    help="Channels to embed into. Examples: 'B', 'G', 'R', 'BG', 'BGR', 'B,G'. Default 'B'.")
    pe.add_argument("--block", type=int, default=8, help="Block size (default 8).")
    pe.add_argument("--key", default="", help="Secret key for HMAC-based mask (must match at extraction).")
    pe.add_argument("--fit", choices=["host", "none"], default="none",
                    help="Resize watermark to host size. Default 'none' (strict identity).")
    pe.add_argument("--binarize", action="store_true",
                    help="Otsu-threshold watermark (flexible mode).")
    pe.add_argument("--save-fitted-wm", default=None,
                    help="Save the exact watermark used for embedding (PNG).")

    # EXTRACT
    px = sub.add_parser("extract", help="Extract watermark from a stego PNG.")
    px.add_argument("--stego", required=True, help="Path to stego PNG.")
    px.add_argument("--out", required=True, help="Output extracted watermark PNG path.")
    px.add_argument("--channels", default="B",
                    help="Channels to extract from (same syntax as embed). Default 'B'.")
    px.add_argument("--block", type=int, default=8, help="Block size used at embedding (default 8).")
    px.add_argument("--key", default="", help="Key used at embedding (must match).")
    # Tamper verification (optional)
    px.add_argument("--wm", dest="ref_wm_path", default=None,
                    help="Reference original watermark PNG to compute tamper heatmap/mask.")
    px.add_argument("--wm-fit", choices=["host", "none"], default="none",
                    help="Resize reference watermark to stego size if needed. Default 'none'.")
    px.add_argument("--wm-binarize", action="store_true",
                    help="Otsu-threshold reference watermark before comparison.")
    px.add_argument("--tamper-heatmap", dest="tamper_heatmap_path", default=None,
                    help="Path to save grayscale tamper heatmap PNG (0: match, 255: full mismatch).")
    px.add_argument("--tamper-mask", dest="tamper_mask_path", default=None,
                    help="Path to save binary tamper mask PNG (requires --tamper-threshold).")
    px.add_argument("--tamper-threshold", type=float, default=0.0,
                    help="Mismatch fraction in [0,1] to mark a block as tampered for the binary mask.")

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "embed":
        cfg = EmbedConfig(
            block=args.block,
            channels=args.channels,
            key=args.key,
            fit=(args.fit == "host"),
            binarize=args.binarize,
            save_fitted_wm=args.save_fitted_wm,
        )
        psnr_val, outp = embed(args.host, args.wm, args.out, cfg)
        print(f"[OK] Watermarked image saved to: {outp}")
        print(f"PSNR(host, stego): {psnr_val:.2f} dB")

    elif args.cmd == "extract":
        cfg = ExtractConfig(
            block=args.block,
            channels=args.channels,
            key=args.key,
            ref_wm_path=args.ref_wm_path,
            ref_wm_fit=(args.wm_fit == "host"),
            ref_wm_binarize=args.wm_binarize,
            tamper_heatmap_path=args.tamper_heatmap_path,
            tamper_mask_path=args.tamper_mask_path,
            tamper_threshold=args.tamper_threshold,
        )
        outp = extract(args.stego, args.out, cfg)
        print(f"[OK] Extracted watermark saved to: {outp}")


if __name__ == "__main__":
    main()
