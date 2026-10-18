# Line Art Live

Real-time, browser-only edge effects for webcams and uploaded videos. It runs
TEED, a person-tuned TEED, PiDiNet, or Canny locally and composites the result
over the original full-resolution frame. Neural models use ONNX Runtime Web with
WebGPU where possible; frames never leave the browser.

The main experiment is **TEED Person**: a 243 KiB TEED model fine-tuned to
find a person directly, without first running a segmentation model. V3 makes
the person/background silhouette the primary signal, keeps internal clothing,
face, and limb edges at lower weight, and learns from real no-person frames so
that surrounding scene edges are less likely to appear. V4 continues from V3
with synthetic full-body people composited into held-out room scenes.

**Live demo:** [ldenoue.github.io/edge](https://ldenoue.github.io/edge/)

## Detectors and person masks

| Option | Purpose | Runtime notes |
| --- | --- | --- |
| TEED | General semantic edges | 320×320 ONNX; WebGPU with WASM fallback |
| TEED Person V4 Synthetic | Stronger full-body silhouette response from synthetic people in rooms | Absolute calibrated output; default threshold `0.46` |
| TEED Person V3 Silhouette | Prioritizes the person/background contour and suppresses scene edges | Absolute calibrated output; default threshold `0.56` |
| TEED Person V2 | Human-prioritized edges with corrected masks and hard negatives | Same 243 KiB architecture as TEED |
| TEED Person V1 | Original three-epoch person experiment | Kept for direct A/B comparison |
| PiDiNet Small | Fewer, heavier perceptual boundaries | 320×320 ONNX |
| Canny | Classical image edges | Plain JavaScript |
| MediaPipe Multiclass | Default optional person-mask gate | 256×256 multiclass Image Segmenter; GPU with CPU fallback |
| MediaPipe Landscape | Faster legacy person-mask gate | 256×144 binary selfie-segmentation model |
| RVM MobileNetV3 | Optional temporally recurrent matte | FP32 or FP16 ONNX; forced to WASM for correctness |

Enable **Person only** to run the selected segmentation model on the same frame
and suppress edge pixels outside its soft person mask. The first picker option
uses MediaPipe's official `selfie_multiclass_256x256` model. It converts the
background confidence into one person mask, so hair, face/body skin, clothes,
and the rest of the segmented person remain inside the gate. MediaPipe Tasks
runs in video mode with GPU delegation and a CPU fallback; its 256×256 output
is bilinearly aligned to the detector's 320×320 grid. The legacy landscape model and both RVM variants
remain available for direct comparison. Segmenters load only when the toggle is
enabled, and their model/runtime assets are served locally from
`public/mediapipe`.

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

### Dataset source and layout

Training uses the **Crowd Instance-level Human Parsing (CIHP)** dataset
introduced by Gong et al. in *Instance-level Human Parsing via Part Grouping
Network* (ECCV 2018). The official
[CIHP/PGN repository](https://github.com/Engineering-Course/CIHP_PGN) links the
dataset downloads; the
[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_Gong_Instance-level_Human_Parsing_ECCV_2018_paper.pdf)
describes its collection and annotations. CIHP contains 38,280 images split
into 28,280 training, 5,000 validation, and 5,000 test images. We only use the
publicly annotated training and validation splits. Review the dataset's terms
before downloading or redistributing it; the dataset itself is not committed
to this repository.

Extract the download so it has this shape (the root directory name may vary):

```text
instance-level_human_parsing/
├── Training/
│   ├── Images/*.jpg
│   ├── Human_ids/*.png
│   └── Category_ids/*.png
├── Validation/
│   ├── Images/*.jpg
│   ├── Human_ids/*.png
│   └── Category_ids/*.png
└── Testing/Images/*.jpg
```

`Human_ids` assigns each person a non-zero integer instance ID.
`Category_ids` assigns the background or one of 19 semantic human-part labels.
The preparation script does not copy source images; manifests contain absolute
paths to the local extraction.

### Reproduce the manifests

Create a Python environment and fetch the stock TEED implementation/checkpoint:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
git clone --depth 1 https://github.com/xavysp/TEED.git vendor/TEED
```

Build balanced train and validation manifests from a local CIHP download:

```bash
python3 scripts/prepare_cihp_person_edges.py \
  --dataset ~/Downloads/instance-level_human_parsing \
  --negative-ratio 0.25 --seed 1021
```

The default output is `data/cihp_person_edges`. Source images are referenced in
place, and training targets can be generated lazily to conserve disk space. Add
`--cache-targets` only when enough free space is available to materialize all
320x320 target PNGs.

Candidate selection is deterministic and works per annotated instance. An
image is selected when at least one person has a head label, a torso label,
both leg labels, a bounding-box height of at least 50% of the image, an area of
at least 2.5%, and no top/bottom clipping. The union mask and target still
include every person in a selected image, including partially visible secondary
people. Scanning the official splits produces 4,106 training and 732 validation
positive images.

CIHP contains people in every image, so it does not provide genuine no-person
frames. Instead, the script derives safe hard negatives from annotated
background regions:

1. Random candidate crops cover 35–70% of the source width and height and are
   at least 64 pixels in each dimension.
2. A crop is rejected if it or a 2% safety margin contains any non-zero
   `Human_ids` pixel.
3. Up to 40 candidates are tested per image. The candidate with the greatest
   grayscale gradient energy plus `0.15 ×` intensity standard deviation is kept,
   favoring clutter over blank sky or walls.
4. At the default ratio, generation stops after 1,026 training and 183
   validation negatives (25% as many negatives as positives).

The positive manifests are `training.jsonl` and `validation.jsonl`; hard
negatives are written separately as `training_negatives.jsonl` and
`validation_negatives.jsonl`. Use `--negative-ratio 0` to disable them.

### Prepare full-frame COCO negatives

CIHP background crops are useful hard negatives, but they do not teach the
model what an entire frame with no person looks like. V3 also uses images from
the COCO 2017 validation split whose instance annotations contain no `person`
category. COCO is described on the official
[dataset page](https://cocodataset.org/dataset/detection-2017.htm). The images
are downloaded selectively rather than fetching the complete dataset.

Download the compact instance annotation file and verify the exact file used
for this experiment:

```bash
mkdir -p data/coco/annotations
curl -L \
  https://huggingface.co/datasets/pcuenq/coco2017-instances/resolve/main/instances_val2017.json \
  -o data/coco/annotations/instances_val2017.json
echo "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f  data/coco/annotations/instances_val2017.json" \
  | shasum -a 256 -c -
```

Then reproduce the V3 negative manifests:

```bash
python3 scripts/prepare_coco_negatives.py \
  --annotations data/coco/annotations/instances_val2017.json \
  --output data/coco_no_person \
  --train-count 1200 --val-count 300 \
  --candidate-multiplier 1.5 --seed 1021 \
  --download-missing --download-workers 12
```

The script found 2,307 images without an annotated person, downloaded and
texture-scored a deterministic pool of 2,250, then retained 1,200 training and
300 validation images. Texture score is grayscale gradient energy plus
`0.15 ×` intensity standard deviation, which favors challenging structured
scenes over blank walls and sky. “No annotated person” is not a proof that no
human is visible, so visually audit contact sheets or random samples when
rebuilding the dataset. Images, annotations, and generated manifests are
ignored by Git.

### How TEED Person V1 and V2 were trained

The source data is the CIHP instance-level human-parsing dataset. Its training
set is intentionally crowded, so selection happens per annotated person rather
than by image aspect ratio. An image enters the balanced subset when at least
one instance has:

- a visible head and torso;
- both left and right leg labels;
- a bounding-box height of at least 50% of the image;
- at least 2.5% of the image area; and
- no clipping at the top or bottom boundary.

This produced **4,106 training images and 732 validation images**. V2 also uses
**1,026 training and 183 validation hard negatives**. These are high-texture,
mask-verified background crops taken from CIHP images with a safety margin from
every annotated person. Once an image
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
masks. V2 fixes an error in V1 that resized integer instance IDs before
binarizing them, making a person's soft boundary depend on its instance ID.
V2 binarizes first, then resizes. Its augmentations cover horizontal flips,
small random crops and rotations, brightness, contrast, saturation, blur, JPEG
compression, and sensor-like noise.

V2 was fine-tuned end-to-end for ten epochs with AdamW, a cosine-decayed `1e-4`
learning rate, batch size 16, and 320×320 inputs on Apple MPS. Against the held-
out hard negatives at the browser's `0.18` normalized threshold, visible edge
pixels fell from 2.08% for V1 to 1.45% for V2 while person-edge response stayed
effectively unchanged. TEED has only 58,910 parameters, so each fine-tuned
browser model remains 243 KiB with the same inference cost as stock TEED.

Fine-tune the official BIPED epoch-5 checkpoint on the manifests, evaluate V1
against V2, then export the best checkpoint for the browser:

```bash
python3 scripts/train_teed_person_edges.py --epochs 10 --batch-size 16 --workers 4 \
  --output checkpoints/teed_person_v2
python3 scripts/convert_teed.py checkpoints/teed_person_v2/best.pth \
  public/teed-person-v2-320.onnx --source vendor/TEED --size 320
python3 scripts/evaluate_teed_person_edges.py \
  checkpoints/teed_person/best.pth checkpoints/teed_person_v2/best.pth
npm run build
```

On macOS, PyTorch training uses MPS when available. The frozen original TEED
model supplies interior edge targets, CIHP supplies the human masks and
silhouettes, and the loss explicitly penalizes response outside human masks.
Checkpoints, manifests, cached targets, the TEED checkout, and `dist/` are
intentionally ignored by Git. The exported ONNX files under `public/` are the
small deployable artifacts that belong in version control.

### Train the silhouette-prioritized V3 model

V3 starts from the V2 checkpoint but keeps the stock BIPED model as the frozen
teacher. It mixes CIHP background crops with full-frame COCO no-person images,
then deterministically caps the combined training negatives at 1,200 so they
do not overwhelm the 4,106 positive images.

The target reflects this project's priority: the person/background boundary is
full strength, while teacher-derived edges inside a person are only one quarter
strength:

```text
target = max(
  0.25 × normalized_teacher_edges × soft_union_of_human_masks,
  human_silhouette_boundary
)
```

Silhouette pixels receive an additional `3 ×` BCE weight (four times the base
pixel weight in total). Background suppression remains at `0.5`; this avoids
the over-suppression observed in early V3 trials while still teaching blank
person masks. Reproduce the selected run with:

```bash
python3 scripts/train_teed_person_edges.py \
  --epochs 6 --batch-size 16 --workers 4 --learning-rate 2e-5 \
  --checkpoint checkpoints/teed_person_v2/best.pth \
  --teacher-checkpoint vendor/TEED/checkpoints/BIPED/5/5_model.pth \
  --background-weight 0.5 --interior-edge-weight 0.25 \
  --boundary-loss-weight 3 \
  --train-negative-manifest \
    data/cihp_person_edges/training_negatives.jsonl \
    data/coco_no_person/training.jsonl \
  --val-negative-manifest \
    data/cihp_person_edges/validation_negatives.jsonl \
    data/coco_no_person/validation.jsonl \
  --train-negative-limit 1200 \
  --output checkpoints/teed_person_v3_silhouette
```

Epoch 3 was selected by the silhouette-focused validation comparison, rather
than copying `best.pth`, which is selected by the broader aggregate loss.
Evaluate and export that checkpoint with:

```bash
python3 scripts/evaluate_teed_person_edges.py \
  checkpoints/teed_person_v2/best.pth \
  checkpoints/teed_person_v3_silhouette/epoch-3.pth \
  --negative-manifest \
    data/cihp_person_edges/validation_negatives.jsonl \
    data/coco_no_person/validation.jsonl
python3 scripts/convert_teed.py \
  checkpoints/teed_person_v3_silhouette/epoch-3.pth \
  public/teed-person-v3-320.onnx --source vendor/TEED --size 320
npm run build
```

At matched positive-image background leakage of about 4.95%, V2 at absolute
threshold `0.46` reached 41.98% visible-silhouette recall and 4.24% active
pixels on full no-person frames. V3 at threshold `0.56` reached 42.52%
visible-silhouette recall and 3.70% on no-person frames. Its lower interior-edge
recall is deliberate: capacity and loss weight were shifted toward the outer
human contour.

Unlike V1/V2 and stock TEED, V3 does not apply per-frame min/max normalization
in the browser. It uses the raw sigmoid output and an absolute default threshold
of `0.56`, preserving the model's learned confidence so a weak no-person frame
cannot be stretched into visible edges.

### Generate synthetic full-body data for V4

V4 experiments add paired renders made with Blender, MPFB's rigged MakeHuman
body, and curated room photos. The RGB image and person mask use the exact same
camera and pose. Shape, skin tone, painted garment colors, pose, framing, room
crop, and lighting are randomized deterministically. Temporary Blender meshes,
materials, and image textures are removed after every frame so long runs do not
continually grow in memory.

The room backgrounds come from Pixabay's `room` photo search. Create a Pixabay
API key, expose it only through the environment, then download the attributed
source set and apply the recorded manual curation:

```bash
export PIXABAY_API_KEY="..."
python3 scripts/download_pixabay_rooms.py \
  --output data/backgrounds/pixabay_rooms --count 100
python3 scripts/curate_pixabay_rooms.py
```

The downloader never writes the API key to disk. It records each photo's source
page, contributor, tags, dimensions, and Pixabay license URL in `manifest.json`.
The checked curation keeps 74 real room/interior scenes and moves 26 recoverable
rejects to `data/backgrounds/pixabay_rooms_rejected`. Rejections contain a
prominent person or animal, an illustration, or an object close-up that is not
a useful scene. The numbered rejection list corresponds to this exact 100-image
download, and each rejected manifest row records the reason.

Install and enable the official MPFB extension in Blender 4.2 or newer, then
render 1,000 pairs at TEED's native resolution:

```bash
blender --background --python scripts/generate_realistic_people.py -- \
  --output data/teed_person_v4_synthetic \
  --backgrounds data/backgrounds/pixabay_rooms \
  --count 1000 --validation-count 100 --size 320 --seed 4104
```

The checked run contains 900 training pairs and 100 validation pairs. It uses
63 room photos for training and holds out the other 11 photos exclusively for
validation. `training.jsonl` and `validation.jsonl` can be read directly by the
existing TEED training script. `metadata.json` records every shape, pose,
background, and split decision.

The selected V4 experiment continues from the selected V3 checkpoint at a
lower learning rate. It retains the V3 silhouette-focused target and adds
available COCO no-person images without depending on the original local CIHP
download:

```bash
python3 scripts/train_teed_person_edges.py \
  --epochs 4 --batch-size 16 --workers 4 --learning-rate 1e-5 \
  --checkpoint checkpoints/teed_person_v3_silhouette/epoch-3.pth \
  --teacher-checkpoint vendor/TEED/checkpoints/BIPED/5/5_model.pth \
  --background-weight 0.5 --interior-edge-weight 0.25 \
  --boundary-loss-weight 3 \
  --train-manifest data/teed_person_v4_synthetic/training.jsonl \
  --val-manifest data/teed_person_v4_synthetic/validation.jsonl \
  --train-negative-manifest data/coco_no_person/training.jsonl \
  --val-negative-manifest data/coco_no_person/validation.jsonl \
  --train-negative-limit 300 --val-negative-limit 100 \
  --output checkpoints/teed_person_v4_synthetic
```

All four epochs were compared with V3 on the 100 held-out synthetic people and
100 real COCO no-person frames. Epoch 2 was selected for the boundary/background
tradeoff rather than aggregate loss. At roughly matched COCO false-edge rates,
V4 at threshold `0.46` reaches 7.14% silhouette recall and 5.22% positive-frame
background leakage; V3 at threshold `0.54` reaches 5.95% and 3.99%, respectively.
This is a 20% relative silhouette-recall gain with more background leakage, so
V4 is an additional experiment rather than a replacement for V3.

Export the selected checkpoint with:

```bash
python3 scripts/convert_teed.py \
  checkpoints/teed_person_v4_synthetic/epoch-2.pth \
  public/teed-person-v4-320.onnx --source vendor/TEED --size 320
```

The generated images, masks, metadata, and manifests live below `data/` and are
ignored by Git. Re-running with the same seed, background set, Blender version,
and MPFB version reproduces the sampling decisions.

Open the printed localhost URL. Drop a browser-supported video onto the stage or
select **Use webcam**. Webcam access works on localhost or HTTPS.

Use the detector picker to compare:

- **TEED** — the fastest neural option and the default.
- **TEED Person V4 Synthetic** — favors full-body silhouette recall after
  fine-tuning on rendered people in held-out room scenes.
- **TEED Person V3 Silhouette** — emphasizes the outer person contour and uses
  calibrated confidence to suppress no-person scenes.
- **TEED Person V2** — corrected masks, hard negatives, and stronger augmentation.
- **TEED Person V1** — the original model, retained for A/B comparison.
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
