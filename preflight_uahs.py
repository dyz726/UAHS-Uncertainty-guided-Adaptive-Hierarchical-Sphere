"""Synthetic full-rank preflight for the L4-only attention ablation."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from network.sphere_model import build_saliency_model
from train import parser as training_parser


def normalized_kl(prediction, target):
    prediction = prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(prediction.clamp_min(1e-8).log(), target, reduction="batchmean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--output",
        default="log/l4_attention_ablation_preflight.json",
    )
    options = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full rank-6 preflight requires CUDA")

    args = training_parser.parse_args([])
    args.model_type = "uahs"
    args.img_rank = 6
    args.seq_length = options.sequence_length
    args.temporal_window_radius = None
    args.return_aux = False
    args.debug_uahs = False
    args.use_checkpoint = True

    device = torch.device("cuda")
    model = build_saliency_model(args).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vertices = model.hierarchy_l4_l6.fine_vertex_count
    reports = []

    for step in range(options.steps):
        inputs = torch.randn(
            1, options.sequence_length, vertices, 3, device=device
        )
        target = torch.rand(
            1, options.sequence_length, vertices, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = model(inputs)
        loss = normalized_kl(
            prediction.reshape(-1, vertices), target.reshape(-1, vertices)
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        reports.append({
            "step": step + 1,
            "loss": float(loss.detach()),
            "seconds": time.perf_counter() - start,
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated(device) / 1024 ** 3
            ),
            "output_shape": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all()),
        })

    report = {
        "architecture": "L4 two-local plus global, feature upsample, L6 head",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": reports,
    }
    output_dir = os.path.dirname(options.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(options.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
