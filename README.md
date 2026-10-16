# Line Art Live

Real-time, browser-only edge effects for webcams and uploaded videos. It runs
TEED, a person-tuned TEED, PiDiNet, or Canny locally and composites the result
over the original full-resolution frame. Neural models use ONNX Runtime Web with
WebGPU where possible; frames never leave the browser.

The main experiment is **TEED Person**: a 243 KiB TEED model fine-tuned to
retain human silhouettes and interior clothing/face/limb detail while
suppressing edges in the surrounding scene. Unlike the optional MediaPipe and
RVM modes, TEED Person does this in one inference pass without first computing
a segmentation mask.

## Detectors and person masks

| Option | Purpose | Runtime notes |
| --- | --- | --- |
| TEED | General semantic edges | 320×320 ONNX; WebGPU with WASM fallback |
| TEED Person | Human-prioritized edges without a runtime mask | Same 243 KiB architecture as TEED |
| PiDiNet Small | Fewer, heavier perceptual boundaries | 320×320 ONNX |
| Canny | Classical image edges | Plain JavaScript |
| MediaPipe | Optional person-mask gate | Landscape selfie-segmentation model |
| RVM MobileNetV3 | Optional temporally recurrent matte | FP32 or FP16 ONNX; forced to WASM for correctness |

Enable **Person only** to run MediaPipe Selfie Segmentation on the same frame
and suppress edge pixels outside its soft person mask. The segmenter is loaded
only when the toggle is enabled, and its model/runtime assets are served locally
from `public/mediapipe`.

Choose **Person mask · green screen** to inspect the segmentation directly: the
captured person is composited over green while the gated edge result stays on
top. Edge inference, segmentation, and compositing all use one frozen video
frame to prevent temporal drift.

The **Mask threshold** control adjusts the center of a softly feathered person
cutoff. Raise it to reject uncertain background pixels; lower it to retain more
hair and fine limbs. Its default of `0.33` preserves the original cutoff.

The **Person model** picker compares the original MediaPipe landscape model
against the official FP32 and FP16 RVM MobileNetV3 exports. RVM carries four
recurrent tensors from frame to frame, so its alpha matte can use temporal
information; the state resets whenever the video source changes, loops, or seeks
backward. RVM uses ONNX Runtime's WASM backend because mixed WebGPU execution
currently produces incorrect alpha values in the browser; the edge detector can
still use WebGPU. The FP16 download is half the size, but CPU/WASM does not
guarantee faster inference. RVM is GPL-3.0 licensed, which must be considered
before redistribution.

## Run it

```bash
npm install
npm run dev
```

## Prepare CIHP person-edge data

Build balanced train and validation manifests from a local CIHP download:

```bash
python3 scripts/prepare_cihp_person_edges.py
```

The default output is `data/cihp_person_edges`. Source images are referenced in
place, and training targets can be generated lazily to conserve disk space. Add
`--cache-targets` only when enough free space is available to materialize all
320x320 target PNGs.

### How TEED Person was trained

The source data is the CIHP instance-level human-parsing dataset. Its training
set is intentionally crowded, so selection happens per annotated person rather
than by image aspect ratio. An image enters the balanced subset when at least
one instance has:

- a visible head and torso;
- both left and right leg labels;
- a bounding-box height of at least 50% of the image;
- at least 2.5% of the image area; and
- no clipping at the top or bottom boundary.

This produced **4,106 training images and 732 validation images**. Once an image
is selected, the target mask contains *every* annotated person, including
partially visible secondary people, so the model is never taught that another
human is background.

The official BIPED epoch-5 TEED checkpoint is used twice: as the student
initialization and as a frozen teacher. For each image, the soft training target
is conceptually:

```text
target = max(
  normalized_teacher_edges × soft_union_of_human_masks,
  human_silhouette_boundary
)
```

This preserves useful internal human edges while setting scene edges to zero.
All four TEED outputs receive deep supervision with class-balanced binary cross
entropy, and an additional loss penalizes response outside the union of human
masks. Training augments horizontal orientation, brightness, and contrast.

The included checkpoint was fine-tuned end-to-end for three epochs with AdamW,
a `1e-4` learning rate, batch size 16, and 320×320 inputs on Apple MPS. TEED has
only 58,910 parameters, so the fine-tuned browser model remains 243 KiB and has
the same inference cost as stock TEED.

Fine-tune the official BIPED epoch-5 checkpoint on the manifests, then export
the best checkpoint for the browser:

```bash
git clone --depth 1 https://github.com/xavysp/TEED.git vendor/TEED
python3 scripts/train_teed_person_edges.py --epochs 3 --batch-size 16 --workers 0
python3 scripts/convert_teed.py checkpoints/teed_person/best.pth \
  public/teed-person-320.onnx --source vendor/TEED --size 320
```

On macOS, PyTorch training uses MPS when available. The frozen original TEED
model supplies interior edge targets, CIHP supplies the human masks and
silhouettes, and the loss explicitly penalizes response outside human masks.

Open the printed localhost URL. Drop a browser-supported video onto the stage or
select **Use webcam**. Webcam access works on localhost or HTTPS.

Use the detector picker to compare:

- **TEED** — the fastest neural option and the default.
- **TEED Person** — prioritizes human edges without enabling **Person only**.
- **PiDiNet Small** — more detailed perceptual boundaries with a larger runtime cost.
- **Canny** — a dependency-free JavaScript implementation with Gaussian blur,
  Sobel gradients, non-maximum suppression, and hysteresis thresholding.

## TEED model conversion

The checked-in `public/teed-320.onnx` is a 243 KiB export of the MIT-licensed
official TEED checkpoint. To reproduce the conversion:

```bash
git clone https://github.com/xavysp/TEED.git /tmp/TEED
python -m pip install torch onnx
python scripts/convert_teed.py \
  /tmp/TEED/checkpoints/BIPED/5/5_model.pth \
  public/teed-320.onnx \
  --source /tmp/TEED \
  --size 320
```

The graph accepts a BIPED-mean-normalized BGR float32 tensor shaped
`1 × 3 × 320 × 320` and returns the sigmoid of TEED's fused edge map. The client
then applies the per-frame min/max normalization used by TEED's official test
code.

The earlier anime U-Net converter remains in `scripts/convert_model.py` for
reference, but that model is no longer shipped by the demo.

## PiDiNet model conversion

The picker uses the official 184K-parameter PiDiNet Small checkpoint, converted
from pixel-difference convolutions to equivalent vanilla convolutions before
ONNX export:

```bash
git clone https://github.com/hellozhuo/pidinet.git /tmp/pidinet
python scripts/convert_pidinet.py \
  /tmp/pidinet/trained_models/table5_pidinet-small.pth \
  public/pidinet-small-320.onnx \
  --source /tmp/pidinet \
  --variant small \
  --size 320
```

PiDiNet's upstream license begins with a research-purpose/commercial-contact
notice even though the remaining text grants MIT-like permissions. Resolve that
ambiguity with the authors before commercial distribution.

## Limitations

The model sees a square 320×320 resize of each frame. This keeps inference fast,
but very fine details are lost and widescreen frames are distorted during edge
detection before the aligned mask is stretched back over the source. Per-frame
normalization can also make edge intensity shift slightly between scenes.

TEED is copyright its contributors and distributed under the MIT License. See
the [upstream repository](https://github.com/xavysp/TEED) and paper for attribution.
