# Adaptive Multi-Resolution Prototype

The repository supports both the original `SphereUFormer` baseline and an
`AdaptiveSphereUFormer` research prototype. Select the prototype with
`--model_type adaptive_sphere_uformer`; the default remains the baseline.

The audited adaptive model keeps SphereUFormer's vertex feature backbone but
defines adaptive regions on exact triangular-face hierarchies. Fine vertex
features are mean/max pooled through face descendants for the coarse encoder.
Coarse vertex features are aggregated to faces, where separate saliency,
heteroscedastic Laplace uncertainty, and refinement heads operate. The
refinement head uses content, predicted uncertainty, and optional temporal
motion. Coarse face gates propagate through exact 1-to-4 descendants and are
averaged over incident faces to gate fine vertex residuals.

The current fine branch still evaluates every fine icosphere vertex. This is a
differentiable dense prototype and **does not implement sparse FLOP reduction**.
Future sparse selection can replace `AdaptiveRegionSelector` without changing
the face-region interface.

## UAHS-V2 Recursive Hierarchy

Select V2 with `--model_type adaptive_sphere_uformer_v2`. For the default
`img_rank=6`, V2 uses adjacent ranks 4→5→6. Rank-5 and rank-6 candidates each
consume RGB features sampled directly from the original rank-6 input; lower
rank upsampling supplies context only. Rank-5-to-6 effective gates are the
product of the exact parent gate and the local child gate. All three encoders
remain dense, so V2 also makes no sparse-computation or FLOP-reduction claim.

```bash
python train.py --model_type adaptive_sphere_uformer_v2 \
  --img_rank 6 --adaptive_coarse_depth 2 \
  --adaptive_middle_depth 1 --adaptive_fine_depth 1 \
  --coarse_pool_type mean_max \
  --target_refine_ratio_l1 0.25 \
  --target_refine_ratio_l2 0.125 \
  --lambda_coarse 0.3 --lambda_uncertainty 0.1 \
  --lambda_refine 0.2 --lambda_budget 0.05
```

V2 refinement targets are binary, spherical-area-aware selections supervised
with BCE logits. L1 selects the most difficult 25% of rank-4 faces. L2 selects
12.5% of global area only from children of the selected L1 parents.

Example training selection (add the normal dataset and optimization options):

```bash
python train.py --model_type adaptive_sphere_uformer \
  --coarse_rank_offset 2 --adaptive_coarse_depth 2 \
  --adaptive_fine_depth 1 --adaptive_region_type face \
  --coarse_pool_type mean_max --lambda_uncertainty 0.1 \
  --target_refine_ratio 0.25
```

Use `--disable_adaptive_refinement` for the uniform fine-refinement ablation,
`--disable_uncertainty_refinement` to remove uncertainty from the refinement
decision, `--disable_motion_refinement` to remove motion, and
`--disable_budget_regularization` to remove the gate-budget loss.

`uncertainty` is an inference-time network prediction trained with a Laplace
negative log-likelihood. The detached coarse saliency error is used separately
as a relative fixed-budget refinement target; it is not called uncertainty.

Run the low-rank CPU check in the project environment with:

```bash
/home/dyz/anaconda3/envs/sphereformer/bin/python smoke_test_adaptive.py
```

Run the geometry audit with:

```bash
/home/dyz/anaconda3/envs/sphereformer/bin/python diagnose_sphere_hierarchy.py
```

Evaluate a trained adaptive checkpoint with streaming UAHS diagnostics:

```bash
/home/dyz/anaconda3/envs/sphereformer/bin/python inference.py \
  --model_type adaptive_sphere_uformer \
  --base_model_weights /path/to/model.pth \
  --dataset_name Sports-360 --dataset_root_dir /path/to/Sports-360 \
  --adaptive_diagnostics --compare_uniform_gate \
  --diagnostics_output /tmp/adaptive_diagnostics.json
```

The JSON report contains mean/std/min/max for predicted uncertainty,
refinement score, fine-face gate, and area-weighted refinement ratio. It also
reports gate saturation fractions, uncertainty/error scale ratio and correlation,
coarse-upsampled versus final KLD, and an optional gate=1 evaluation of the
same checkpoint. A positive `adaptive_gain_positive_is_better` means the
adaptive prediction outperformed its uniform-gate counterpart. Diagnostics
only consume existing auxiliary outputs and do not alter the model or loss.

For V2, use `--compare_gate_baselines` to evaluate `full_fine`,
`uniform_same_budget`, `random_same_budget`, and `oracle_error_selection`.
Level-4 and level-5 reports include uncertainty Pearson/Spearman correlations,
scale/error ratio, selection IoU/precision/recall, and actual refined area.
