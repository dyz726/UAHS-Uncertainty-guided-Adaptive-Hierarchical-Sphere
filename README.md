# L4 Attention Ablation — Sports-360 Saliency

This project is an ablation copy of `Sampling_test`. The `uahs` model keeps
only the rank-4 attention backbone:

1. project the rank-6 RGB input and pool it directly to rank 4;
2. run exactly two local motion-aware attention blocks at rank 4;
3. run one full-sphere global spatio-temporal attention block at rank 4;
4. interpolate the rank-4 **features** to rank 5 and then rank 6; and
5. apply the only saliency output head at rank 6.

There is no L4 saliency head, rank-5/rank-6 attention, uncertainty estimation,
hard routing, dynamic budget, or auxiliary loss. Training supervises only the
final rank-6 saliency map `[B, T, 40962]` when `--img_rank 6` is used.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python /home/dyz/PythonProject/Test_Codes/sports_test/train.py \
  --model_type uahs \
  --dataset_name Sports-360 \
  --dataset_root_dir /home/dyz/PythonProject/Dataset/Sports-360 \
  --mode vertex \
  --img_rank 6 \
  --seq_length 12 \
  --temporal_window_radius none \
  --coarse_pool_type mean_max \
  --global_query_chunk_size 128 \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --num_workers 8 \
  --num_epochs 100 \
  --optimizer adamw \
  --learning_rate 1e-4 \
  --min_learning_rate 1e-6 \
  --warmup_epochs 3 \
  --lr_scheduler warmup_cosine \
  --weight_decay 1e-4 \
  --use_checkpoint 1 \
  --accum_grads 1 \
  --exp_name l4-attention-ablation-sports360 \
  --log_dir /home/dyz/PythonProject/Test_Codes/sports_test/log/l4-attention-ablation-sports360 \
  --tensorboard_log_dir /home/dyz/PythonProject/log/tensorboard/l4-attention-ablation-sports360
```

`--model_type sphere_uformer` still selects the copied, unchanged baseline.
The legacy uncertainty/budget CLI options remain accepted for command-line
compatibility but do not create modules or contribute losses in this ablation.

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python /home/dyz/PythonProject/Test_Codes/sports_test/inference.py \
  --model_type uahs \
  --dataset_name Sports-360 \
  --dataset_root_dir /home/dyz/PythonProject/Dataset/Sports-360 \
  --base_model_weights /path/to/l4_ablation_checkpoint.pth \
  --output_dir /home/dyz/PythonProject/DataSet_Output/Sports-360 \
  --method_name L4-Attention-Ablation \
  --mode vertex \
  --img_rank 6 \
  --seq_length 12 \
  --temporal_window_radius none
```

Inference requires a checkpoint matching this ablation architecture. A full
UAHS checkpoint is intentionally rejected because it contains removed heads
and refinement modules. Training can still use `--base_model_weights` to load
only shape-compatible parameters as initialization.

## Verification

```bash
python -m compileall .
/home/dyz/anaconda3/envs/sphereformer/bin/python smoke_test_uahs.py
```

## VR-EyeTracking

Run the following commands from this project directory. The loader reads
`train_list.txt` and `test_list.txt` (only `--dataset_split 1`).
The official 134 training videos are split into 107 training and 27 validation
videos; the 74 test videos remain held out.

```bash
python train.py --model_type uahs --dataset_name VR-EyeTracking \
  --dataset_root_dir /home/dyz/PythonProject/Dataset/VR-EyeTracking \
  --dataset_split 1 --img_rank 6 --seq_length 12 \
  --log_dir /path/to/vr-training-log

python inference.py --model_type uahs --dataset_name VR-EyeTracking \
  --dataset_root_dir /home/dyz/PythonProject/Dataset/VR-EyeTracking \
  --dataset_split 1 --img_rank 6 --seq_length 12 \
  --base_model_weights /path/to/this-project-checkpoint.pth \
  --output_dir /path/to/vr-predictions --save_mat
```

Video frames, maps and fixations are matched by the original zero-based frame
number. A clip missing either annotation is skipped without renumbering later
clips. Sequential decode timestamps verify random seeks; the first run builds
a small index cache under `~/.cache/uahs/vr_frame_pts` (or `XDG_CACHE_HOME`).
No full-resolution frames are cached.

Training/validation use complete clips. Test/inference pad a final short clip
but count and save only its real frames. PNGs retain source frame names and
are reconstructed at a fixed width of 256 and height of 128; MAT files contain
`frame_indices` alongside `salmap`.
Use checkpoints for this project's own model architecture.

`train.py --test --base_model_weights ...` reports sphere metrics on real test
frames only. `inference.py` additionally saves predictions and computes ERP
metrics. To evaluate existing PNGs independently, supply the actual
`saliency_png` directory and the same clip length used for inference:

```bash
python evaluate/evaluation.py --dataset_name VR-EyeTracking \
  --dataset_root_dir /home/dyz/PythonProject/Dataset/VR-EyeTracking \
  --saliency_folder /path/to/saliency_png --seq_length 12
```

Run the frame alignment, missing-label, padding and evaluation regressions with
`python -m unittest -q test_vr_eyetracking_loader`.
