# UAHS — Uncertainty-guided Adaptive Hierarchical Sphere

This repository contains the published `SphereUFormer` baseline and the final
UAHS model for 360° video saliency prediction. UAHS maps rank-6 RGB video to a
rank-6 spherical saliency map `[B, T, 40962]` through six components:

1. local motion-aware spherical modeling at rank 4 (two blocks);
2. true full-sphere, content-aware reasoning at rank 4 (one block);
3. learned Laplace uncertainty at ranks 4 and 5;
4. uncertainty-ranked hard spherical-area selection at 25% and 12.5%;
5. selected-query sparse refinement at ranks 5 and 6; and
6. rank-4/rank-5/rank-6 multi-exit reconstruction.

The rank-5 and rank-6 refiners gather only selected query vertices and their
fixed spherical neighbors. They do not run dense high-resolution spatial
attention. Upsampling supplies context only; fine detail is pooled/projected
from the original rank-6 observation.

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

The UAHS objective is final saliency loss plus rank-4/rank-5 saliency and
heteroscedastic Laplace uncertainty losses. The predicted uncertainty directly
ranks each hard area selector; labels never enter `forward()`.

## Verification

Run geometry, hard-budget, sparse-equivalence, no-label-leak, gradient, and
baseline regression tests:

```bash
/home/dyz/anaconda3/envs/sphereformer/bin/python smoke_test_uahs.py
```

Run the real V100 FP32 preflight (`B=1`, `T=12`, rank 4→5→6, three optimizer
steps) and write its memory/timing report:

```bash
CUDA_VISIBLE_DEVICES=0 /home/dyz/anaconda3/envs/sphereformer/bin/python \
  preflight_uahs.py --output log/uahs_uncertainty_only_preflight.json
```

## Evaluation and Diagnostics

```bash
python inference.py --model_type uahs \
  --base_model_weights /path/to/uahs.pth \
  --dataset_root_dir /path/to/Sports-360 --metrics_only \
  --uahs_diagnostics \
  --selector_comparison_modes saliency_score random_same_budget \
    oracle_error_same_budget
```

Diagnostics include uncertainty calibration/correlation, selector
IoU/precision/recall, hierarchical KL, exact selected spherical area, active
vertices/queries, and estimated refinement work. Oracle error selection is a
diagnostic reference only and is never used in normal inference.
