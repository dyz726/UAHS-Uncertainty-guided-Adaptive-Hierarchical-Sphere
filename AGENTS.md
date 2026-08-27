# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python/PyTorch implementation of SphereUFormer for 360-degree video saliency prediction. `train.py` defines the command-line configuration and launches training through `train_salient.py`; `inference.py` loads checkpoints, writes saliency maps, and runs evaluation. Model components live in `network/`, dataset implementations and loader factories in `data/`, and ERP evaluation utilities in `evaluate/`. Shared spherical geometry and saliency metrics are in `trimesh_utils.py` and `Sphere_SalientScore_torch.py`. Keep generated checkpoints, TensorBoard logs, prediction PNGs, and datasets outside version-controlled source directories.

## Build, Test, and Development Commands

There is no build step or dependency manifest. Use a dedicated Python environment and install the imported scientific stack (PyTorch, OpenCV, NumPy, SciPy, trimesh, einops, tqdm, TensorBoard, and Weights & Biases) with versions compatible with your CUDA setup.

- `python -m compileall .` performs a fast syntax/import-compilation check.
- `python train.py --dataset_name Sports-360 --dataset_root_dir /path/to/data --no_gpu --num_epochs 1 --ltr 1` runs a minimal CPU training smoke test.
- `python train.py --dataset_name Sports-360 --dataset_root_dir /path/to/data --test --base_model_weights /path/to/model.pth` runs the trainer's test path.
- `python inference.py --dataset_root_dir /path/to/data --base_model_weights /path/to/model.pth --output_dir /tmp/predictions --max_batches 1` validates inference on one batch.

Use `python train.py --help` or `python inference.py --help` for all model and dataset options.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, modules, and CLI flags; use `PascalCase` for classes. Keep tensor shape assumptions explicit in docstrings or nearby comments, and preserve device/dtype when creating tensors. Prefer repository-relative imports and configurable paths over new machine-specific absolute paths. No formatter or linter is configured, so keep changes focused and match surrounding style.

## Testing Guidelines

No automated test suite or coverage threshold is currently configured. For every change, run `compileall` plus a one-batch smoke test relevant to the edited path. Data-loader changes should cover each affected dataset (`Sports-360`, `AVS-ODV`, or `SVGC_AVA`); model changes should verify output shapes and finite saliency metrics.

## Commit & Pull Request Guidelines

History uses short, direct subjects such as `Update saliency training code`; follow that style and keep each commit scoped to one concern. Pull requests should describe the dataset and configuration used, list validation commands, link related issues, and report metric or tensor-shape impact. Include representative output images when prediction or visualization behavior changes, but never commit datasets, checkpoints, credentials, or experiment logs.
