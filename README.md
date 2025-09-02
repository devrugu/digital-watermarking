# Digital Watermarking (Ping Wah Wong–inspired)

A clean, production-ready Python implementation of **block-based image watermarking** (PNG in/out) that is color-capable, reproducible, and simple to use from both the CLI and Python API.

By default the tool runs in **strict identity** mode: if your `watermark.png` has the **same size** as `host.png` and is already **binary** (0/255), the **extracted watermark is pixel-identical** to the watermark you embedded. You can also opt into **flexible** behavior (auto-resize + Otsu binarization), **multi-channel redundancy** (B/G/R) with majority voting at extraction, and **tamper visualization** (heatmap + binary mask).

---

## Features

- **PNG in, PNG out** (lossless). Alpha channel is preserved.
- **Color-capable**: embed/extract in any subset of `B`, `G`, `R`. Grayscale is supported too.
- **Strict identity (default)**: extracted watermark equals the input watermark **bit-for-bit** when sizes match and the watermark is binary (0/255).
- **Flexible mode**: `--fit host` (resize) and/or `--binarize` (Otsu) for non-matching, non-binary watermarks. You can **save the exact bits** used at embed time (`--save-fitted-wm`) for auditing.
- **Tamper visualization**: per-block mismatch **heatmap** and **binary mask** against a provided reference watermark.
- **Multi-channel redundancy**: embed the same watermark in multiple channels; extraction uses **majority vote** across channels for robustness.
- **Professional logs**: timestamped logs, adjustable verbosity (`--log-level`), and progress reporting (`--progress`).

---

## Algorithm (concise)

This implementation follows the spirit of **Ping Wah Wong's** verification approach, adapted for practical capacity (1 LSB/pixel). For each 8×8 block on each selected channel:

1. Zero LSBs → X̃_r
2. Compute a keyed mask  
   P_r = HMAC-SHA256(X̃_r || block coords || image size, key)[0:64]
3. Take 64 watermark bits B_r from the watermark image (strict or fitted)
4. Compute W_r = B_r ⊕ P_r and **write W_r** into the block's **LSBs**

Extraction reverses the operation to recover B_r. With multiple channels selected, the same B_r is embedded in each channel and recovered by **majority vote**.

**Tamper map**: if you provide the original watermark at extraction, we compute the per-block mismatch ratio between the extracted bits and the reference; this yields a **heatmap** (0 → match, 255 → full mismatch) and an optional binary **mask** via a threshold.

---

## Security model & limitations

- This is **tamper-evident watermarking** bound to block content via a keyed HMAC. It **does not encrypt** or hide the watermark; it makes tampering detectable.
- LSB schemes are **not** robust to **lossy compression, rescaling, color-space conversions**, or heavy filtering. Stick to **lossless** workflows (PNG).
- Keep your **key** secret. Anyone with the key can embed or extract valid watermarks.
- Public-key signatures can be layered on top by distributing a digital signature across blocks; this repo focuses on a practical, high-capacity variant.

---

## Requirements & install

```bash
# Python 3.9+ recommended
pip install -U numpy opencv-python pytest
````

---

## Quick start

**Strict identity** (recommended for demos): the watermark must have the **same H×W** as the host and be **binary** (0/255).

```bash
# Embed
python watermarking.py embed \
  --host images/host.png \
  --wm images/watermark.png \
  --out outputs/stego.png \
  --key "demo" \
  --channels B

# Extract
python watermarking.py extract \
  --stego outputs/stego.png \
  --out outputs/extracted.png \
  --key "demo" \
  --channels B

# Optional, Linux/macOS: verify equality (should report IDENTICAL)
cmp images/watermark.png outputs/extracted.png && echo "IDENTICAL"
```

---

## CLI usage

### Embedding

```bash
python watermarking.py embed --host HOST --wm WM --out OUT
                            [--channels B|G|R|BG|BGR] [--block 8]
                            [--key KEY]
                            [--fit host|none] [--binarize]
                            [--save-fitted-wm PATH]
                            [--log-level LEVEL] [--progress N]
```

| Flag               | Default | Description                                                                      |
| ------------------ | ------- | -------------------------------------------------------------------------------- |
| `--host`           | –       | Host PNG path.                                                                   |
| `--wm`             | –       | Watermark PNG path.                                                              |
| `--out`            | –       | Output stego PNG path.                                                           |
| `--channels`       | `B`     | Any combination of `B`, `G`, `R` (e.g., `BGR` or `B,G`). Redundancy if multiple. |
| `--block`          | `8`     | Block size (use 8×8 for classic behavior).                                       |
| `--key`            | `""`    | Secret key for the HMAC-based mask. Must match at extraction.                    |
| `--fit`            | `none`  | If `host`, resize watermark to host size (flexible mode).                        |
| `--binarize`       | off     | If set, Otsu-threshold the watermark (flexible mode).                            |
| `--save-fitted-wm` | –       | Save the exact watermark bits used (binary PNG) for auditing.                    |
| `--log-level`      | `INFO`  | `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                             |
| `--progress`       | `10`    | Progress step in percent; `0` disables.                                          |

> **Strict identity** is active when **both** `--fit none` and `--binarize` are omitted.

### Extraction

```bash
python watermarking.py extract --stego STEGO --out OUT
                              [--channels B|G|R|BG|BGR] [--block 8]
                              [--key KEY]
                              [--wm REF] [--wm-fit host|none] [--wm-binarize]
                              [--tamper-heatmap HEAT.png]
                              [--tamper-mask MASK.png] [--tamper-threshold 0.125]
                              [--log-level LEVEL] [--progress N]
```

| Flag                                   | Default | Description                                                                  |
| -------------------------------------- | ------- | ---------------------------------------------------------------------------- |
| `--stego`                              | –       | Stego PNG path.                                                              |
| `--out`                                | –       | Output extracted watermark PNG path (binary 0/255).                          |
| `--channels`                           | `B`     | Channels to extract from; majority vote if multiple.                         |
| `--block`                              | `8`     | Block size used at embedding.                                                |
| `--key`                                | `""`    | Key used at embedding.                                                       |
| `--wm` / `--wm-fit` / `--wm-binarize`  | –       | Provide the original watermark; processed similarly for **tamper analysis**. |
| `--tamper-heatmap`                     | –       | Save grayscale per-block mismatch (0 match, 255 full mismatch).              |
| `--tamper-mask` + `--tamper-threshold` | –       | Save binary mask: blocks with mismatch fraction ≥ threshold are flagged.     |
| `--log-level`                          | `INFO`  | Logging verbosity.                                                           |
| `--progress`                           | `10`    | Progress step in percent; `0` disables.                                      |

---

## Examples

**Multi-channel redundancy + majority vote**

```bash
python watermarking.py embed \
  --host images/host.png \
  --wm images/watermark.png \
  --out outputs/stego.png \
  --key "demo" --channels BGR

python watermarking.py extract \
  --stego outputs/stego.png \
  --out outputs/extracted.png \
  --key "demo" --channels BGR
```

**Flexible:** resize + binarize non-binary watermark; save fitted bits

```bash
python watermarking.py embed \
  --host images/host.png \
  --wm images/watermark.png \
  --out outputs/stego.png \
  --key "demo" --channels BGR \
  --fit host --binarize \
  --save-fitted-wm outputs/used_wm.png

python watermarking.py extract \
  --stego outputs/stego.png \
  --out outputs/extracted.png \
  --key "demo" --channels BGR
# Now: extracted.png == used_wm.png
```

**Tamper heatmap + mask**

```bash
python watermarking.py extract \
  --stego outputs/stego.png \
  --out outputs/extracted.png \
  --key "demo" --channels BGR \
  --wm images/watermark.png --wm-fit host --wm-binarize \
  --tamper-heatmap outputs/heatmap.png \
  --tamper-mask outputs/tamper_mask.png --tamper-threshold 0.125
```

---

## Python API

```python
import watermarking as wm

# Embed (strict identity)
psnr, out_path = wm.embed(
    host_png_path="images/host.png",
    wm_png_path="images/watermark.png",
    out_png_path="outputs/stego.png",
    cfg=wm.EmbedConfig(channels="B", key="demo")
)

# Extract (majority vote if multiple channels were used)
out_wm = wm.extract(
    stego_png_path="outputs/stego.png",
    out_wm_png_path="outputs/extracted.png",
    cfg=wm.ExtractConfig(channels="B", key="demo")
)
```

---

## Project layout

```
digital-watermarking/
├─ images/                  # sample inputs (you can keep your own here)
│  ├─ host.png
│  └─ watermark.png
├─ outputs/                 # generated artifacts (stego/extracted/heatmaps) – gitignored
├─ tests/                   # pytest minisuite
│  ├─ __init__.py
│  ├─ conftest.py
│  ├─ test_cli_smoke.py
│  ├─ test_fit_binarize.py
│  ├─ test_identity.py
│  └─ test_multichannel_and_tamper.py
├─ watermarking.py          # main module + CLI
├─ README.md
└─ LICENSE
```

---

## Testing

```bash
pytest -q
```

The suite covers strict identity, flexible fitting/binarization, multi-channel voting, tamper outputs, PSNR sanity, and a CLI smoke test.

---

## Performance notes

* Time is linear in pixel count; memory is modest (processing by blocks).
* Default **8×8** blocks give 1 LSB/pixel capacity with good visual quality (PSNR typically > 38 dB for natural images).
* For very large images, use `--log-level INFO` and a larger `--progress` step or `0` to reduce console overhead.

---

## License

See [LICENSE](LICENSE).