# UAHS — Uncertainty-guided Adaptive Hierarchical Sphere

This repository contains the published `SphereUFormer` baseline and the final
UAHS model for 360° video saliency prediction. UAHS maps rank-6 RGB video to a
rank-6 spherical saliency map `[B, T, 40962]` through six components:

1. local motion-aware spherical modeling at rank 4 (two blocks);
2. true full-sphere, content-aware reasoning at rank 4 (one block);
3. learned Laplace uncertainty at ranks 4 and 5;
4. uncertainty-ranked hard spherical-area selection at fixed configured ratios;
5. selected-query sparse spatial refinement at ranks 5 and 6; and
6. rank-4/rank-5/rank-6 multi-exit reconstruction.

The rank-5 and rank-6 refiners gather only selected query vertices and their
fixed spherical neighbors. They do not run dense high-resolution spatial
attention. Temporal attention remains in the rank-4 coarse backbone only.
Upsampling supplies context only; fine detail is pooled/projected
from the original rank-6 observation. With `--abs_pos_enc_in 1` (the default),
independent input position encodings are added at ranks 4, 5, and 6 before
coarse modeling or sparse refinement. Fine-level spatial attention also uses
its own positional Q/K encoding and relative bias.

## Training

The final model is selected with `--model_type uahs` (the default):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type uahs --dataset_name Sports-360 \
  --dataset_root_dir /path/to/Sports-360 \
  --img_rank 6 --seq_length 12 --temporal_window_radius none \
  --train_batch_size 1 --val_batch_size 1 \
  --target_refine_ratio_l1 0.25 --target_refine_ratio_l2 0.125
```

Use `--model_type sphere_uformer` for the unchanged baseline. Run
`python train.py --help` for loss weights and optimization options.

The two `target_refine_ratio` values are fixed global spherical-area budgets.
Uncertainty ranks each hard selector; labels never enter `forward()`. At ranks 5
and 6, only selected spatial queries are refined.

Earlier fixed- or dynamic-budget checkpoints can initialize compatible shared
training weights. Direct inference requires a checkpoint trained with the
current architecture.

### VR-EyeTracking

Use the dataset's `train_list.txt` and `test_list.txt` with the shared
`videos/<video_id>`, `maps/<video_id>/<frame>.png`, and
`fixation/<video_id>/<frame>.png` layout. Frame numbers are zero-based. A clip
is skipped if any of its frames lacks either annotation; later clips retain
their original video-frame indices. Inference pads a final short clip for the
model and writes predictions only for its real frames. On first use, the loader
scans each video once to index decoded-frame timestamps (cached under
`~/.cache/uahs/vr_frame_pts`); this corrects OpenCV seek offsets without
extracting or storing full-resolution frames.

```bash
python train.py --model_type uahs --dataset_name VR-EyeTracking \
  --dataset_root_dir /path/to/VR-EyeTracking --img_rank 6 --seq_length 12

python inference.py --model_type uahs --dataset_name VR-EyeTracking \
  --dataset_root_dir /path/to/VR-EyeTracking \
  --base_model_weights /path/to/uahs.pth \
  --output_dir /path/to/predictions --img_rank 6 --seq_length 12
```

Prediction PNG names preserve the annotation frame numbers and use the
annotation image dimensions. With `--save_mat`, `frame_indices` in each MAT
file identifies the source video frame for every stored prediction.

## Verification

Run geometry, fixed-budget, sparse-equivalence, no-label-leak, gradient, and
baseline regression tests:

```bash
/home/dyz/anaconda3/envs/sphereformer/bin/python smoke_test_uahs.py
```

Run the real V100 FP32 preflight (`B=1`, `T=12`, rank 4→5→6, three optimizer
steps) and write its memory/timing report:

```bash
CUDA_VISIBLE_DEVICES=0 /home/dyz/anaconda3/envs/sphereformer/bin/python \
  preflight_uahs.py --output log/uahs_spatial_only_preflight.json
```

## Evaluation and Diagnostics

```bash
python inference.py --model_type uahs \
  --base_model_weights /path/to/uahs.pth \
  --dataset_root_dir /path/to/Sports-360 --metrics_only \
  --uahs_diagnostics
```

Diagnostics include uncertainty calibration/correlation, selector
IoU/precision/recall, hierarchical KL, exact selected spherical area, active
vertices/queries, estimated refinement work, and the final uncertainty-routing
metrics. Inference always evaluates only the formal uncertainty route.
