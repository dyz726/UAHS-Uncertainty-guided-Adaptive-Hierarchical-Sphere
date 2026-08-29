import os            
import time          
import warnings        
import math
from itertools import islice
import tqdm
import wandb              
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from adaptive_objectives import build_fixed_area_target
from data.get_saliency_dataloaders import get_dataloaders         
from network.sphere_model import build_saliency_model
from Sphere_SalientScore_torch import *
from trimesh_utils import *
EPS = 2.2204e-16
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
        )

                  
        self.config = dict(
            img_rank=args.img_rank,
            img_width=args.img_width,
            node_type=args.mode,
            num_scales=args.num_scales,
            win_size_coef=args.win_size_coef,
            scale_factor=args.scale_factor,
            downsample=args.downsample,
            scale_depth=args.scale_depth,
            model_type=args.model_type,
            coarse_pool_type=getattr(args, "coarse_pool_type", "mean_max"),
            target_refine_ratio_l1=args.target_refine_ratio_l1,
            target_refine_ratio_l2=args.target_refine_ratio_l2,
            global_query_chunk_size=args.global_query_chunk_size,
            hard_selection_warmup_epochs=args.hard_selection_warmup_epochs,
            lambda_saliency_l4=args.lambda_saliency_l4,
            lambda_saliency_l5=args.lambda_saliency_l5,
            lambda_uncertainty_l4=args.lambda_uncertainty_l4,
            lambda_uncertainty_l5=args.lambda_uncertainty_l5,
            lambda_refine_l4=args.lambda_refine_l4,
            lambda_refine_l5=args.lambda_refine_l5,
            use_uncertainty_refinement=getattr(
                args, "use_uncertainty_refinement", True
            ),
            use_motion_refinement=args.use_motion_refinement,
        )

                    
        self.wandb_run = None
        os.makedirs(args.log_dir, exist_ok=True)

        self.writer = None
        if args.enable_tensorboard:
            tb_log_dir = args.tensorboard_log_dir or os.path.join(args.log_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            print("TensorBoard log dir:", tb_log_dir)
        if args.wandb_project:
            self.wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.exp_name,
                group=args.wandb_group,
                dir=args.log_dir,
            )
            self._run = wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{self.wandb_run.id}")

                 
        self.loader_train, self.loader_val = get_dataloaders(
            is_test=args.test,
            dataset_name=args.dataset_name,
            dataset_root_dir=args.dataset_root_dir,
            dataset_kwargs={
                "sphere_rank": args.img_rank,
                "grid_width": args.img_width,
                "sphere_node_type": self.config["node_type"],
                "seq_length": args.seq_length
            },
            augmentation_kwargs=dict(
                color_augmentation=False,
                lr_flip_augmentation=False,
                yaw_rotation_augmentation=False,
            ),
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            dataset_split=args.avs_split,
        )
                                      
        self.model = build_saliency_model(args, node_type=self.config["node_type"])
                                       
        if args.load_weights_task and not args.base_model_weights:
            raise ValueError("--load_weights_task requires --base_model_weights")
        if args.base_model_weights:
            self.model = self.load_pretrained(self.model, args.base_model_weights)

                
        total_params = sum(p.numel() for p in self.model.parameters())
        if self.wandb_run:
            wandb.log({f"total_params": total_params}, step=0)

                 
        self.model.to(self.device)

                              
        self.val_mean_loss = []
        self.train_mean_loss = []
                
                                                          
                                         

        if args.accum_grads <= 0:
            raise ValueError("--accum_grads must be positive")
        if args.warmup_epochs < 0:
            raise ValueError("--warmup_epochs must be non-negative")
        if not 0 <= args.min_learning_rate <= args.learning_rate:
            raise ValueError("--min_learning_rate must be between 0 and --learning_rate")
        if (
                args.temporal_window_radius is not None
                and args.temporal_window_radius < 0
        ):
            raise ValueError(
                "--temporal_window_radius must be non-negative or None"
            )
        target_ratio_l1 = args.target_refine_ratio_l1
        target_ratio_l2 = args.target_refine_ratio_l2
        if not 0 <= target_ratio_l2 <= target_ratio_l1 <= 1:
            raise ValueError(
                "Expected 0 <= target_refine_ratio_l2 <= "
                "target_refine_ratio_l1 <= 1"
            )
        loss_weights = (
            args.lambda_saliency_l4,
            args.lambda_saliency_l5,
            args.lambda_uncertainty_l4,
            args.lambda_uncertainty_l5,
            args.lambda_refine_l4,
            args.lambda_refine_l5,
        )
        if min(loss_weights) < 0:
            raise ValueError("UAHS loss weights must be non-negative")

        param_groups = self.get_optimizer_param_groups(args.weight_decay)
        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=args.learning_rate)
        elif args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups, lr=args.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        self.base_learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        requested_batches = args.limit_train_batches
        if math.isfinite(requested_batches):
            self.train_batches_per_epoch = min(len(self.loader_train), int(requested_batches))
        else:
            self.train_batches_per_epoch = len(self.loader_train)
        if self.train_batches_per_epoch <= 0:
            raise ValueError("No training batches are available")
        self.steps_per_epoch = math.ceil(self.train_batches_per_epoch / args.accum_grads)
        self.total_optimizer_steps = max(1, self.steps_per_epoch * args.num_epochs)
        self.warmup_steps = min(
            self.total_optimizer_steps,
            max(0, int(args.warmup_epochs * self.steps_per_epoch)),
        )
        self.scheduler = None
        if args.lr_scheduler == "reduce_on_plateau":
                                                                             
                                                                        
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                threshold=0.0,
                threshold_mode="abs",
                min_lr=args.min_learning_rate,
            )

                 
        self.parameters_to_train = list(self.model.parameters())
                
        print("Training is using: ", self.device)
        print("Total parameters: ", total_params)
        print(
            "Optimizer:", args.optimizer,
            "LR scheduler:", args.lr_scheduler,
            "base_lr:", self.base_learning_rate,
            "min_lr:", self.min_learning_rate,
            "warmup_steps:", self.warmup_steps,
            "total_steps:", self.total_optimizer_steps,
        )
                                                               

                      
                  
        self.icosphere_ref = IcoSphereRef(self.config["node_type"])
        normals = self.icosphere_ref.get_normals(rank=self.config["img_rank"])
        normals_rphitheta = asSpherical(normals)
        self.normals_wh = np.stack(
            (
                normals_rphitheta[:, 2] / 180,                          
                normals_rphitheta[:, 1] / 180 * 2 - 1,                       
            ),
            axis=1,
        ).astype(np.float32)          
        self.normals_wh_tensor = torch.from_numpy(self.normals_wh).to(self.device)
              
        if self.wandb_run:
            self.wandb_run.config.update(self.config)

    def get_optimizer_param_groups(self, weight_decay):
        no_decay_names = set()
        if hasattr(self.model, "no_weight_decay"):
            no_decay_names.update(self.model.no_weight_decay())
        no_decay_keywords = set()
        if hasattr(self.model, "no_weight_decay_keywords"):
            no_decay_keywords.update(self.model.no_weight_decay_keywords())
        no_decay_keywords.update({"bias_grid", "logit_scale"})

        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or name in no_decay_names
                or any(keyword in name for keyword in no_decay_keywords)
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def get_learning_rate(self, optimizer_step: int) -> float:
        if self.args.lr_scheduler != "warmup_cosine":
            return self.optimizer.param_groups[0]["lr"]

        if self.warmup_steps > 0 and optimizer_step < self.warmup_steps:
            return self.base_learning_rate * float(optimizer_step + 1) / self.warmup_steps

        cosine_steps = max(1, self.total_optimizer_steps - self.warmup_steps)
        cosine_step = min(max(optimizer_step - self.warmup_steps, 0), cosine_steps)
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_steps))
        return self.min_learning_rate + (
            self.base_learning_rate - self.min_learning_rate
        ) * cosine_ratio

    def update_learning_rate(self) -> float:
        learning_rate = self.get_learning_rate(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        return learning_rate

    def load_pretrained(self, model, pretrained_path):
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"Model weights not found: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        source_parameter_count = len(pretrained_dict)
        model_dict = model.state_dict()

                    
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
                     
        match_keys = [k for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape]
        print(
            f"Loaded {len(match_keys)}/{len(model_dict)} model tensors "
            f"from {source_parameter_count} tensors in {pretrained_path}"
        )

                 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    def reconstruct_erp_torch(self, sphere_sal, target_size):
        """
        将球面数据重建为ERP图像 - PyTorch向量化加速版本
        sphere_sal: 球面显著性值 [L] 或 [B, L] (numpy数组或torch张量)
        target_size: 目标尺寸 (H, W)
        """
        H, W = target_size
        device = self.device

                   
        if isinstance(sphere_sal, np.ndarray):
                                   
            sphere_sal = torch.from_numpy(sphere_sal).to(device)

                 
        if sphere_sal.dim() == 1:
            sphere_sal = sphere_sal.unsqueeze(0)                 

        B, L = sphere_sal.shape

                    
        cache_key = f"grid_{H}_{W}"
        if not hasattr(self, '_precomputed_grids'):
            self._precomputed_grids = {}

        if cache_key not in self._precomputed_grids:
                              
            scale_factor = torch.tensor([(W - 1) / 2, (H - 1) / 2],
                                        device=device, dtype=torch.float32)
            coords_pixel = (self.normals_wh_tensor + 1) * scale_factor

                       
            x, y = coords_pixel[:, 0], coords_pixel[:, 1]
            x0, y0 = torch.floor(x).long(), torch.floor(y).long()
            x1, y1 = x0 + 1, y0 + 1

                             
            x0 = torch.clamp(x0, 0, W - 1)
            y0 = torch.clamp(y0, 0, H - 1)
            x1 = torch.clamp(x1, 0, W - 1)
            y1 = torch.clamp(y1, 0, H - 1)

                  
            dx = x - x0.float()
            dy = y - y0.float()
            w00 = (1 - dx) * (1 - dy)
            w10 = dx * (1 - dy)
            w01 = (1 - dx) * dy
            w11 = dx * dy

                     
            self._precomputed_grids[cache_key] = {
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'w00': w00, 'w10': w10, 'w01': w01, 'w11': w11
            }

                 
        grid = self._precomputed_grids[cache_key]
        x0, y0, x1, y1 = grid['x0'], grid['y0'], grid['x1'], grid['y1']
        w00, w10, w01, w11 = grid['w00'], grid['w10'], grid['w01'], grid['w11']

                          
        erp_imgs = torch.zeros((B, H, W), device=device, dtype=sphere_sal.dtype)
        weight_imgs = torch.zeros((B, H, W), device=device, dtype=sphere_sal.dtype)

                   
                                
        batch_indices = torch.arange(B, device=device)[:, None].expand(B, L)

                        
        erp_imgs.index_put_(
            (batch_indices, y0, x0),
            w00 * sphere_sal,
            accumulate=True
        )
        weight_imgs.index_put_(
            (batch_indices, y0, x0),
            w00,
            accumulate=True
        )

                        
        erp_imgs.index_put_(
            (batch_indices, y0, x1),
            w10 * sphere_sal,
            accumulate=True
        )
        weight_imgs.index_put_(
            (batch_indices, y0, x1),
            w10,
            accumulate=True
        )

                        
        erp_imgs.index_put_(
            (batch_indices, y1, x0),
            w01 * sphere_sal,
            accumulate=True
        )
        weight_imgs.index_put_(
            (batch_indices, y1, x0),
            w01,
            accumulate=True
        )

                        
        erp_imgs.index_put_(
            (batch_indices, y1, x1),
            w11 * sphere_sal,
            accumulate=True
        )
        weight_imgs.index_put_(
            (batch_indices, y1, x1),
            w11,
            accumulate=True
        )

               
        weight_imgs = torch.clamp(weight_imgs, min=1e-8)
        result = erp_imgs / weight_imgs

                         
        if result.shape[0] == 1:
            result = result.squeeze(0)
                            

        result = result.cpu().numpy()
        return result

    def inputs_to_device(self, inputs):
        """将输入数据移动到指定设备"""
        device_inputs = {}
        for k, v in inputs.items():
            if "rgb" in k:
                                 
                device_inputs[k] = v.to(self.device)
            elif "sal" in k:
                              
                device_inputs[k] = v.to(self.device)
            elif "fix" in k:
                device_inputs[k] = v.to(self.device)
        return device_inputs

    def test(self):
        """测试模式入口"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        try:
            self.validate()
        finally:
            if self.writer:
                self.writer.close()

    def train(self):
        """主训练循环"""
        self.epoch = 0
        self.mini_step = 0
        self.step = 0
        self.start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_learning_rate()

              
                         

        try:
            for self.epoch in range(1, self.args.num_epochs + 1):
                self.train_one_epoch()
                current_val_loss = self.validate()

                if self.scheduler is not None:
                    previous_lr = self.optimizer.param_groups[0]["lr"]
                    self.scheduler.step(current_val_loss)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr < previous_lr:
                        print(f"Learning rate reduced: {previous_lr:.3e} -> {current_lr:.3e}")
                    if self.writer:
                        self.writer.add_scalar("val/learning_rate", current_lr, self.epoch)
                        self.writer.flush()

                if self.args.enable_save and self.epoch % self.args.save_frequency == 0:
                    self.save_model()
        finally:
            if self.writer:
                self.writer.close()

    def loss_kl_cc(self, pre_sal, gt_sal,gt_fix):
                    
                    
                     
                                          
                                                         
                                                         
                                                            
                              
                              
                                        
                                      
         
                                                                                                
                                                       
        return kl_sphere_torch(pre_sal,gt_sal)

    @staticmethod
    def area_weighted_mean(values, face_areas):
        weights = face_areas.to(
            device=values.device, dtype=values.dtype
        )
        weights = weights / weights.sum()
        return (values * weights.reshape(1, 1, -1)).sum(dim=-1).mean()

    @staticmethod
    def area_weighted_masked_mean(values, face_areas, mask):
        """Average within each sample's eligible spherical parent area."""
        weights = face_areas.to(
            device=values.device, dtype=values.dtype
        ).reshape(1, 1, -1)
        weights = weights * mask.to(dtype=values.dtype)
        denominator = weights.sum(dim=-1)
        numerator = (values * weights).sum(dim=-1)
        per_sample = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-8),
            torch.zeros_like(numerator),
        )
        return per_sample.mean()

    def compute_uahs_losses(self, ground_truth, outputs):
        """Compute hard-hierarchy auxiliary losses; labels never enter forward."""
        B, T = ground_truth.shape[:2]
        target_l4 = self.model.aggregate_img_values_to_l4_faces(ground_truth)
        target_l5 = self.model.aggregate_img_values_to_l5_faces(ground_truth)
        saliency_l4 = outputs["saliency_l4"]
        saliency_l5 = outputs["saliency_l5"]
        loss_saliency_l4 = self.loss_kl_cc(
            saliency_l4.reshape(B * T, -1),
            target_l4.reshape(B * T, -1),
            gt_fix=None,
        ) / (B * T)
        loss_saliency_l5 = self.loss_kl_cc(
            saliency_l5.reshape(B * T, -1),
            target_l5.reshape(B * T, -1),
            gt_fix=None,
        ) / (B * T)

        uncertainty_l4 = outputs["uncertainty_l4"]
        uncertainty_l5 = outputs["uncertainty_l5"]
        laplace_l4 = (
            (target_l4 - saliency_l4).abs() / uncertainty_l4
            + torch.log(uncertainty_l4)
        )
        laplace_l5 = (
            (target_l5 - saliency_l5).abs() / uncertainty_l5
            + torch.log(uncertainty_l5)
        )
        loss_uncertainty_l4 = self.area_weighted_mean(
            laplace_l4, self.model.hierarchy_l4_l5.coarse_face_areas
        )
        loss_uncertainty_l5 = self.area_weighted_mean(
            laplace_l5, self.model.hierarchy_l5_l6.coarse_face_areas
        )

        difficulty_l4 = (target_l4 - saliency_l4.detach()).abs()
        target_ratio_l1 = self.args.target_refine_ratio_l1
        target_ratio_l2 = self.args.target_refine_ratio_l2
        selection_target_l4 = build_fixed_area_target(
            difficulty_l4,
            self.model.hierarchy_l4_l5.coarse_face_areas,
            target_ratio_l1,
        )
        eligible_l5 = self.model.hierarchy_l4_l5.propagate_coarse_face_values(
            selection_target_l4
        ).bool()
        difficulty_l5 = (target_l5 - saliency_l5.detach()).abs()
        selection_target_l5 = build_fixed_area_target(
            difficulty_l5,
            self.model.hierarchy_l5_l6.coarse_face_areas,
            target_ratio_l2,
            eligible_mask=eligible_l5,
        )

        refine_bce_l4 = F.binary_cross_entropy_with_logits(
            outputs["refine_logits_l4"],
            selection_target_l4,
            reduction="none",
        )
        refine_bce_l5 = F.binary_cross_entropy_with_logits(
            outputs["refine_logits_l5"],
            selection_target_l5,
            reduction="none",
        )
        loss_refine_l4 = self.area_weighted_mean(
            refine_bce_l4, self.model.hierarchy_l4_l5.coarse_face_areas
        )
        loss_refine_l5 = self.area_weighted_masked_mean(
            refine_bce_l5,
            self.model.hierarchy_l5_l6.coarse_face_areas,
            eligible_l5,
        )
        return {
            "loss_saliency_l4": loss_saliency_l4,
            "loss_saliency_l5": loss_saliency_l5,
            "loss_uncertainty_l4": loss_uncertainty_l4,
            "loss_uncertainty_l5": loss_uncertainty_l5,
            "loss_refine_l4": loss_refine_l4,
            "loss_refine_l5": loss_refine_l5,
            "selection_target_l4": selection_target_l4,
            "selection_target_l5": selection_target_l5,
        }

    def train_one_epoch(self):
        """训练单个epoch"""
        self.model.train()
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(self.epoch)
        batches = islice(self.loader_train, self.train_batches_per_epoch)
        pbar = tqdm.tqdm(batches, total=self.train_batches_per_epoch)
        pbar.set_description(f"## {self.args.exp_name} ## Training Epoch_{self.epoch}")
        loss_sum = []
        sal_metrics = {'AUC': [], 'NSS': [], 'CC': [], 'SIM': [], 'KL': []}
        for batch_idx, inputs in enumerate(pbar, start=1):
            self.mini_step += 1
            inputs_ = self.inputs_to_device(inputs)
            outputs, losses = self.process_batch(inputs_,inputs["videoID"])
            loss_sum.append(losses["loss"].item())

            group_start = ((batch_idx - 1) // self.args.accum_grads) * self.args.accum_grads
            group_size = min(
                self.args.accum_grads,
                self.train_batches_per_epoch - group_start,
            )
            (losses["loss"] / group_size).backward()

            should_step = (
                batch_idx % self.args.accum_grads == 0
                or batch_idx == self.train_batches_per_epoch
            )
            if should_step:
                self.optimizer.step()
                self.step += 1
                learning_rate = self.update_learning_rate()
                self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(
                    lr=f"{learning_rate:.3e}",
                    loss=f"{losses['loss'].item():.4f}",
                )

                                
            with torch.no_grad():
                              
                pred_sal = outputs["pred_sal"]
                gt_sal = inputs_["normalized_sphere_sal"]
                gt_fix = inputs_["normalized_sphere_fix"]
                      
                metrics = batch_compute_metrics(pred_sal, gt_sal, gt_fix, self.device)
                for k, v in metrics.items():
                    sal_metrics[k].append(v)
                if batch_idx % 50 == 0:
                    self.save_pre_comparison(inputs,gt_sal,pred_sal,batch_idx,mode="train")

        train_loss = sum(loss_sum) / len(loss_sum)
        train_metrics = {
            name: torch.mean(torch.stack(values)).item()
            for name, values in sal_metrics.items()
            if values
        }
        print("Train Mean Loss:", train_loss)
        print("Train Metrics:", train_metrics)

        if self.writer:
            self.writer.add_scalar("train/loss", train_loss, self.epoch)
            self.writer.add_scalar(
                "train/learning_rate", self.optimizer.param_groups[0]["lr"], self.epoch
            )
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, self.epoch)
            self.writer.flush()

        if self.wandb_run:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{f"train/{name}": value for name, value in train_metrics.items()},
                },
                step=self.epoch,
                commit=False,
            )

    def validate(self):
        """验证模型性能"""
        self.model.eval()
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(self.epoch)
        pbar = tqdm.tqdm(self.loader_val)
        pbar.set_description(f"Validating Epoch_{self.epoch}")
        loss_sum = []
        sal_metrics = {'AUC': [], 'NSS': [], 'CC': [], 'SIM': [], 'KL': []}
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                inputs_ = self.inputs_to_device(inputs)
                outputs, losses = self.process_batch(inputs_,inputs["videoID"])

                loss_sum.append(losses["loss"].item())
                               
                pred_sal = outputs["pred_sal"]                      
                gt_sal = inputs_["normalized_sphere_sal"]                      
                gt_fix = inputs_["normalized_sphere_fix"]                      

                metrics = batch_compute_metrics(pred_sal, gt_sal, gt_fix,self.device)
                for k, v in metrics.items():
                    sal_metrics[k].append(v)
                if batch_idx % 50 == 0:
                    self.save_pre_comparison(inputs,gt_sal,pred_sal,batch_idx,mode="val")

        current_val_loss = sum(loss_sum) / len(loss_sum)
        val_metrics = {
            name: torch.mean(torch.stack(values)).item()
            for name, values in sal_metrics.items()
            if values
        }
        print("Val Mean Loss:", current_val_loss)
        print("Val Metrics:", val_metrics)

        if self.writer:
            self.writer.add_scalar("val/loss", current_val_loss, self.epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f"val/{name}", value, self.epoch)
            self.writer.flush()

        if self.wandb_run:
            wandb.log(
                {
                    "val/loss": current_val_loss,
                    **{f"val/{name}": value for name, value in val_metrics.items()},
                },
                step=self.epoch,
                commit=True,
            )

        return current_val_loss

    def process_batch(self, inputs,videoID=None):
        """处理单个批次数据"""
        x = inputs["normalized_sphere_rgb"]
        gt_sal = inputs["normalized_sphere_sal"]
        gt_fix = inputs["normalized_sphere_fix"]

        is_uahs = self.args.model_type == "uahs"
        if is_uahs:
            uahs_outputs = self.model(x, return_aux=True)
            pred_sal = uahs_outputs["saliency"]
        else:
            uahs_outputs = None
            pred_sal = self.model(x)

              
        B, T, L = gt_sal.shape
        gt_probs = gt_sal.reshape(B * T, L)
        pre_probs = pred_sal.reshape(B * T, L)
        gt_fix = gt_fix.reshape(B * T, L)

                  
        loss_rec = self.loss_kl_cc(pre_probs, gt_probs,gt_fix)/(B*T)

        auxiliary_losses = {}
        total_loss = loss_rec
        if uahs_outputs is not None:
            auxiliary_losses = self.compute_uahs_losses(gt_sal, uahs_outputs)
            total_loss = (
                loss_rec
                + self.args.lambda_saliency_l4
                * auxiliary_losses["loss_saliency_l4"]
                + self.args.lambda_saliency_l5
                * auxiliary_losses["loss_saliency_l5"]
                + self.args.lambda_uncertainty_l4
                * auxiliary_losses["loss_uncertainty_l4"]
                + self.args.lambda_uncertainty_l5
                * auxiliary_losses["loss_uncertainty_l5"]
                + self.args.lambda_refine_l4
                * auxiliary_losses["loss_refine_l4"]
                + self.args.lambda_refine_l5
                * auxiliary_losses["loss_refine_l5"]
            )

        outputs = {"pred_sal": pred_sal.detach()}
        losses = {
            "loss": total_loss,
            "loss_saliency": loss_rec,
        }
        losses.update({
            name: value
            for name, value in auxiliary_losses.items()
            if name.startswith("loss_")
        })
        return outputs, losses

    def save_model(self):
        """保存模型和优化器状态"""
        save_folder = os.path.join(self.wandb_run.dir, "models") if self.wandb_run else os.path.join(self.args.log_dir,
                                                                                                     "models")
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving model at {save_folder}")

              
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, os.path.join(save_folder, "Epoch_" + str(self.epoch) + "model.pth"))

    def save_pre_comparison(self,inputs,gt_sal,pred_sal,batch_idx,mode="train"):
        sample_idx = 0          
        frame_idx = 0        

        frame_files = inputs["seq_path"][0][0]          

                
        H_orig, W_orig = 128, 256

                
        pred_sal_sample = pred_sal[sample_idx, frame_idx].detach().cpu().numpy()
        pred_erp = self.reconstruct_erp_torch(pred_sal_sample, (H_orig, W_orig))

                
        pred_erp = (pred_erp - pred_erp.min()) / (pred_erp.max() - pred_erp.min() + 1e-8)
        pred_erp = (pred_erp * 255).astype(np.uint8)
        save_dir = os.path.join(self.args.log_dir, mode+"predict", f"epoch_{self.epoch}")
        os.makedirs(save_dir, exist_ok=True)

                  
                                                 
        sal_erp = gt_sal[sample_idx,frame_idx].detach().cpu().numpy()
        sal_erp = self.reconstruct_erp_torch(sal_erp, (H_orig, W_orig))
        sal_erp = (sal_erp - sal_erp.min()) / (sal_erp.max() - sal_erp.min() + 1e-8)
        sal_erp = (sal_erp * 255).astype(np.uint8)

              
        font_scale = 0.5           
        thickness = 1         
        pred_erp_color = cv2.putText(
            pred_erp, "Prediction", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
        gt_erp_color = cv2.putText(
            sal_erp, "Ground Truth", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )

              
        concat_img = cv2.hconcat([gt_erp_color, pred_erp_color])
                            
        line_width = 3      
        line_color = (255, 255, 255)
        concat_img = cv2.line(
            concat_img,
            (W_orig, 0),             
            (W_orig, H_orig - 1),             
            line_color,
            line_width
        )
        cv2.imwrite(os.path.join(save_dir, f"batch_{batch_idx}_comparison.png"), concat_img)
                            
