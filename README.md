# Adaptive Multi-Resolution Prototype

The repository supports both the original `SphereUFormer` baseline and an
`AdaptiveSphereUFormer` research prototype. Select the prototype with
`--model_type adaptive_sphere_uformer`; the default remains the baseline.

The adaptive model performs low-resolution global encoding, predicts a
motion-aware refinement score, and applies a differentiable soft gate to a
shallow fine-resolution branch. The current fine branch still evaluates every
fine icosphere node. This first version is intended to test adaptive
multi-resolution refinement and **does not implement sparse FLOP reduction**.
Future sparse selection can replace `AdaptiveRegionSelector` without changing
the coarse/fine feature interface.

Example training selection (add the normal dataset and optimization options):

```bash
python train.py --model_type adaptive_sphere_uformer \
  --coarse_rank_offset 2 --adaptive_coarse_depth 2 \
  --adaptive_fine_depth 1 --target_refine_ratio 0.25
```

Use `--disable_adaptive_refinement` for the uniform fine-refinement ablation,
`--disable_motion_refinement` for content-only scores, and
`--disable_budget_regularization` to remove the gate-budget loss.

Run the low-rank CPU check in the project environment with:

```bash
/home/dyz/anaconda3/envs/sphereformer/bin/python smoke_test_adaptive.py
```
