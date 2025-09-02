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

Features
--------
- **Multi-channel embedding/extraction** over any subset of {B,G,R}.
  We embed the SAME watermark in each selected channel (redundancy);
  extraction combines with **majority vote** for robustness.
- **Tamper heatmap** (and optional binary mask) when the original
  watermark is provided at extraction; highlights blocks with mismatches.
- **Professional logs** with `--log-level` and `--progress` percent step.

Algorithm (practical variant of Wong)
-------------------------------------
Per 8×8 block on each selected channel:
  1) Zero LSBs to form X̃_r.
  2) Compute P_r = HMAC-SHA256(X̃_r || coords || size, key)[0:64 bits].
  3) Form W_r = B_r XOR P_r, where B_r are 64 watermark bits.
  4) Write W_r into the block's LSBs.
Extraction reverses these steps to recover B_r exactly.

CLI snippets
------------
Strict identity:
    python watermarking.py embed --host host.png --wm watermark.png --out stego.png --key K --channels B
    python watermarking.py extract --stego stego.png --out extracted.png --key K --channels B

Flexible (resize + binarize + save fitted wm, multi-channel):
    python watermarking.py embed --host host.png --wm watermark.png --out stego.png \
        --key K --channels BGR --fit host --binarize --save-fitted-wm used_wm.png

Tamper heatmap:
    python watermarking.py extract --stego stego.png --out extracted.png --key K --channels BGR \
        --wm watermark.png --wm-fit host --wm-binarize \
        --tamper-heatmap heatmap.png --tamper-mask tamper_mask.png --tamper-threshold 0.125
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np


# --------------------------- Logging ------------------------------------- #

LOGGER = logging.getLogger("watermarking")

def _configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(lvl)

def _log_binariness(bits: np.ndarray, name: str) -> None:
    ones = int(np.sum(bits))
    total = bits.size
    pct = 100.0 * ones / total if total else 0.0
    LOGGER.debug("%s: ones=%d / %d (%.2f%%)", name, ones, total, pct)

def _progress_enabled(step_percent: int) -> bool:
    return isinstance(step_percent, int) and step_percent > 0

def _maybe_report_progress(kind: str, chan_label: str, k: int, total: int,
                           step_percent: int, last_pct: int) -> int:
    """Report progress in fixed percent steps; return updated last percentage."""
    if not _progress_enabled(step_percent):  # disabled
        return last_pct
    if total <= 0:
        return last_pct
    pct = int((k * 100) // total)
    if pct >= last_pct + step_percent:
        LOGGER.info("[%s][%s] %d%% (%d/%d blocks)", kind, chan_label, pct, k, total)
        return pct
    return last_pct


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
        LOGGER.info("Resizing watermark to host size: %dx%d -> %dx%d", wm_gray.shape[1], wm_gray.shape[0], W, H)
        wm_gray = cv2.resize(wm_gray, (W, H), interpolation=cv2.INTER_AREA)
    if binarize:
        LOGGER.info("Binarizing watermark using Otsu thresholding")
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
    progress_step: int = 10 # percent step for progress logs; 0 disables


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
    progress_step: int = 10        # percent step for progress logs; 0 disables


def _parse_channels(ch_str: str, color_ndim: int) -> List[int]:
    """
    Parse channel selection string into BGR indices.
    Accepts: 'B', 'G', 'R', 'BG', 'BGR', 'B,G', case-insensitive.
    If grayscale image, returns [-1] sentinel meaning single plane.
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
    LOGGER.info("Preparing watermark bits (target %dx%d, fit=%s, binarize=%s)", W, H, fit, binarize)
    wm_bits = _binary_from_png(wm_src, (H, W), fit=fit, binarize=binarize)
    if not fit:
        h0, w0 = _to_gray(wm_src).shape[:2]
        if (h0, w0) != (H, W):
            raise ValueError(
                f"Strict identity requires watermark size == host size ({H}x{W}); got {h0}x{w0}. "
                f"Use --fit host to resize."
            )
    _log_binariness(wm_bits, "Watermark bits")
    return wm_bits


def embed(host_png_path: str, wm_png_path: str, out_png_path: str,
          cfg: EmbedConfig = EmbedConfig()) -> Tuple[float, str]:
    """
    Embed a watermark into the host image across selected channels.
    Returns (psnr_value, out_path).
    """
    LOGGER.info("Embedding start")
    LOGGER.info("Host: %s | Watermark: %s | Out: %s", host_png_path, wm_png_path, out_png_path)

    host = cv2.imread(host_png_path, cv2.IMREAD_UNCHANGED)
    if host is None:
        raise FileNotFoundError(f"Host image not found: {host_png_path}")
    host = _ensure_uint8(host)
    color, alpha = _split_color_and_alpha(host)
    LOGGER.info("Host image shape: %s (alpha=%s)", str(color.shape), "yes" if alpha is not None else "no")

    # Select channels
    ch_idxs = _parse_channels(cfg.channels, color.ndim)
    ch_labels = ["GRAY"] if ch_idxs == [-1] else [{0: "B", 1: "G", 2: "R"}[ci] for ci in ch_idxs]
    LOGGER.info("Channels selected for embedding: %s", ",".join(ch_labels))
    LOGGER.info("Mode: %s", "STRICT IDENTITY" if (not cfg.fit and not cfg.binarize) else "FLEXIBLE")

    # Set up planes list
    if ch_idxs == [-1]:
        planes = [color.copy()]
    else:
        planes = [color[:, :, ci].copy() for ci in ch_idxs]

    H, W = planes[0].shape[:2]
    wm_bits = _prepare_wm_bits((H, W), wm_png_path, fit=cfg.fit, binarize=cfg.binarize)

    # Optionally save the exact watermark used
    if cfg.save_fitted_wm:
        cv2.imwrite(cfg.save_fitted_wm, (wm_bits * 255).astype(np.uint8))
        LOGGER.info("Saved fitted watermark to: %s", cfg.save_fitted_wm)

    key_bytes = cfg.key.encode("utf-8")
    LOGGER.info("Block size: %d | Key bytes: %d", cfg.block, len(key_bytes))

    # Embed SAME bits into each selected plane
    embedded_planes = []
    for plane, label in zip(planes, ch_labels):
        plane_padded, orig_hw = _pad_to_block(plane, cfg.block)
        wm_padded, _ = _pad_to_block(wm_bits, cfg.block)
        h_pad, w_pad = plane_padded.shape
        blocks_x = w_pad // cfg.block
        blocks_y = h_pad // cfg.block
        total_blocks = blocks_x * blocks_y
        LOGGER.info("[EMBED][%s] blocks: %d x %d = %d", label, blocks_x, blocks_y, total_blocks)

        k = 0
        last_pct = -1
        for by in range(0, h_pad, cfg.block):
            for bx in range(0, w_pad, cfg.block):
                blk = plane_padded[by:by + cfg.block, bx:bx + cfg.block].copy()
                blk_zeroed = (blk & 0xFE).astype(np.uint8)
                wm_blk = wm_padded[by:by + cfg.block, bx:bx + cfg.block].astype(np.uint8).flatten()
                pr = _pr_bits(blk_zeroed, (bx // cfg.block, by // cfg.block), (h_pad, w_pad), key_bytes)
                wr = (wm_blk ^ pr).reshape(cfg.block, cfg.block)
                blk_zeroed[wr == 1] |= 0x01
                plane_padded[by:by + cfg.block, bx:bx + cfg.block] = blk_zeroed

                # progress
                k += 1
                last_pct = _maybe_report_progress("EMBED", label, k, total_blocks, cfg.progress_step, last_pct)

        embedded_planes.append(_crop_to_shape(plane_padded, orig_hw))
        LOGGER.info("[EMBED][%s] done", label)

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

    val_psnr = psnr(host, out_img)
    LOGGER.info("Saved stego: %s", out_png_path)
    LOGGER.info("PSNR(host, stego): %.2f dB", val_psnr)
    LOGGER.info("Embedding completed")
    return val_psnr, out_png_path


def _extract_bits_from_plane(plane: np.ndarray, block: int, key_bytes: bytes,
                             label: str, progress_step: int) -> np.ndarray:
    plane_padded, orig_hw = _pad_to_block(plane, block)
    h_pad, w_pad = plane_padded.shape
    blocks_x = w_pad // block
    blocks_y = h_pad // block
    total_blocks = blocks_x * blocks_y
    LOGGER.info("[EXTRACT][%s] blocks: %d x %d = %d", label, blocks_x, blocks_y, total_blocks)

    wm_rec = np.zeros_like(plane_padded, dtype=np.uint8)

    k = 0
    last_pct = -1
    for by in range(0, h_pad, block):
        for bx in range(0, w_pad, block):
            blk = plane_padded[by:by + block, bx:bx + block].copy()
            lsbs = (blk & 0x01).flatten().astype(np.uint8)
            blk_zeroed = (blk & 0xFE).astype(np.uint8)
            pr = _pr_bits(blk_zeroed, (bx // block, by // block), (h_pad, w_pad), key_bytes)
            wm_blk = (pr ^ lsbs).reshape(block, block)  # 0/1
            wm_rec[by:by + block, bx:bx + block] = wm_blk

            # progress
            k += 1
            last_pct = _maybe_report_progress("EXTRACT", label, k, total_blocks, progress_step, last_pct)

    LOGGER.info("[EXTRACT][%s] done", label)
    return _crop_to_shape(wm_rec, orig_hw)  # 0/1


def _majority_vote(bitmaps: List[np.ndarray]) -> np.ndarray:
    """Majority vote across K bitmaps of shape HxW (values 0/1)."""
    stack = np.stack(bitmaps, axis=0).astype(np.int16)  # KxHxW
    votes = np.sum(stack, axis=0)
    return (votes >= (stack.shape[0] + 1) // 2).astype(np.uint8)


def extract(stego_png_path: str, out_wm_png_path: str, cfg: ExtractConfig = ExtractConfig()) -> str:
    """
    Extract watermark from stego image. If multiple channels are selected,
    majority vote is applied to combine them (unless only one is selected).
    If `ref_wm_path` is provided, a tamper heatmap/mask can be written.
    """
    LOGGER.info("Extraction start")
    LOGGER.info("Stego: %s | Out: %s", stego_png_path, out_wm_png_path)

    stego = cv2.imread(stego_png_path, cv2.IMREAD_UNCHANGED)
    if stego is None:
        raise FileNotFoundError(f"Stego image not found: {stego_png_path}")
    stego = _ensure_uint8(stego)
    color, _ = _split_color_and_alpha(stego)
    LOGGER.info("Stego image shape: %s", str(color.shape))

    ch_idxs = _parse_channels(cfg.channels, color.ndim)
    ch_labels = ["GRAY"] if ch_idxs == [-1] else [{0: "B", 1: "G", 2: "R"}[ci] for ci in ch_idxs]
    LOGGER.info("Channels selected for extraction: %s", ",".join(ch_labels))
    LOGGER.info("Block size: %d", cfg.block)

    # Build per-channel planes list
    if ch_idxs == [-1]:
        planes = [color.copy()]
    else:
        planes = [color[:, :, ci].copy() for ci in ch_idxs]

    key_bytes = cfg.key.encode("utf-8")

    # Extract per channel
    per_channel_bits = [
        _extract_bits_from_plane(p, cfg.block, key_bytes, lbl, cfg.progress_step)
        for p, lbl in zip(planes, ch_labels)
    ]

    # Combine (vote) if multiple
    if len(per_channel_bits) == 1:
        final_bits = per_channel_bits[0]
        LOGGER.info("Single channel used; skipping majority vote")
    else:
        LOGGER.info("Applying majority vote across %d channels", len(per_channel_bits))
        final_bits = _majority_vote(per_channel_bits)

    # Save final extraction
    if not cv2.imwrite(out_wm_png_path, (final_bits * 255).astype(np.uint8)):
        raise IOError(f"Failed to write output PNG: {out_wm_png_path}")
    LOGGER.info("Saved extracted watermark: %s", out_wm_png_path)

    # Tamper heatmap/mask if reference watermark is provided
    if cfg.ref_wm_path:
        H, W = final_bits.shape[:2]
        LOGGER.info("Computing tamper map vs provided reference watermark: %s", cfg.ref_wm_path)
        ref_bits = _prepare_wm_bits((H, W), cfg.ref_wm_path, fit=cfg.ref_wm_fit, binarize=cfg.ref_wm_binarize)

        # Per-block mismatch ratio
        block = cfg.block
        heat = np.zeros_like(final_bits, dtype=np.uint8)
        mismatched_blocks = 0
        total_blocks = 0
        for by in range(0, H, block):
            for bx in range(0, W, block):
                fblk = final_bits[by:by+block, bx:bx+block]
                rblk = ref_bits[by:by+block, bx:bx+block]
                # guard for edge blocks
                hh = min(fblk.shape[0], rblk.shape[0], block)
                ww = min(fblk.shape[1], rblk.shape[1], block)
                fblk = fblk[:hh, :ww]
                rblk = rblk[:hh, :ww]
                denom = float(hh * ww) if (hh and ww) else 1.0
                mism = np.sum(fblk != rblk) / denom
                val = int(round(255.0 * mism))
                heat[by:by+hh, bx:bx+ww] = val
                total_blocks += 1
                if cfg.tamper_threshold > 0.0 and mism >= cfg.tamper_threshold:
                    mismatched_blocks += 1

        if cfg.tamper_heatmap_path:
            cv2.imwrite(cfg.tamper_heatmap_path, heat)
            LOGGER.info("Saved tamper heatmap: %s (0=match, 255=full mismatch)", cfg.tamper_heatmap_path)

        if cfg.tamper_mask_path and cfg.tamper_threshold > 0.0:
            mask = (heat >= int(round(255.0 * cfg.tamper_threshold))).astype(np.uint8) * 255
            cv2.imwrite(cfg.tamper_mask_path, mask)
            pct_blocks = 100.0 * mismatched_blocks / total_blocks if total_blocks else 0.0
            LOGGER.info("Saved tamper mask: %s | threshold=%.3f | flagged=%d/%d blocks (%.2f%%)",
                        cfg.tamper_mask_path, cfg.tamper_threshold, mismatched_blocks, total_blocks, pct_blocks)

    LOGGER.info("Extraction completed")
    return out_wm_png_path


# ------------------------------ CLI ------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Block-LSB watermarking (PNG in/out), color-capable, identity-safe.")
    p.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR. Default INFO.")
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
    pe.add_argument("--progress", type=int, default=10,
                    help="Progress step in percent for logs (0 disables). Default 10.")

    # EXTRACT
    px = sub.add_parser("extract", help="Extract watermark from a stego PNG.")
    px.add_argument("--stego", required=True, help="Path to stego PNG.")
    px.add_argument("--out", required=True, help="Output extracted watermark PNG path.")
    px.add_argument("--channels", default="B",
                    help="Channels to extract from (same syntax as embed). Default 'B'.")
    px.add_argument("--block", type=int, default=8, help="Block size used at embedding (default 8).")
    px.add_argument("--key", default="", help="Key used at embedding (must match).")
    px.add_argument("--progress", type=int, default=10,
                    help="Progress step in percent for logs (0 disables). Default 10.")
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
    _configure_logging(args.log_level)

    if args.cmd == "embed":
        cfg = EmbedConfig(
            block=args.block,
            channels=args.channels,
            key=args.key,
            fit=(args.fit == "host"),
            binarize=args.binarize,
            save_fitted_wm=args.save_fitted_wm,
            progress_step=args.progress,
        )
        psnr_val, outp = embed(args.host, args.wm, args.out, cfg)
        LOGGER.info("[OK] Watermarked image saved to: %s", outp)
        LOGGER.info("PSNR(host, stego): %.2f dB", psnr_val)

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
            progress_step=args.progress,
        )
        outp = extract(args.stego, args.out, cfg)
        LOGGER.info("[OK] Extracted watermark saved to: %s", outp)


if __name__ == "__main__":
    main()
