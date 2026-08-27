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
