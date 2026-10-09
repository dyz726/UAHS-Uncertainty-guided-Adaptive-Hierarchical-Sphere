import torch

EPS = 2.2204e-16


def auc_judd_sphere_torch(pred_sal, gt_fix,device):
    """球面顶点版本的AUC Judd (PyTorch实现)"""
    gt_fix = gt_fix > 0.5         
    if not torch.any(gt_fix):
        return torch.tensor(float('nan'))
                 
    pred_s = pred_sal.float() + torch.rand(pred_sal.shape, device=device) * 1e-7

         
    pred_s = (pred_s - torch.min(pred_s))/ (torch.max(pred_s)- torch.min(pred_s) + EPS)

    fix_sal = pred_s[gt_fix]
    n_fix = fix_sal.numel()
    n_pixels = pred_s.numel()

                                                     
                                               
    sorted_asc = torch.sort(pred_s)[0]
    thresholds = torch.sort(fix_sal, descending=True)[0]
    rank = torch.searchsorted(sorted_asc, thresholds)
    above_th = (n_pixels - rank).float()

             
    k = torch.arange(1, n_fix + 1, device=pred_s.device, dtype=torch.float32)
    zeros = torch.zeros(1, device=pred_s.device)
    ones = torch.ones(1, device=pred_s.device)
    tp = torch.cat([zeros, k / n_fix, ones])
    fp = torch.cat([zeros, (above_th - k) / (n_pixels - n_fix), ones])

                  
    auc = torch.trapz(tp, fp)
    return auc


def nss_sphere_torch(pred_sal, gt_fix):
    """球面顶点版本的NSS (PyTorch实现)"""
    gt_fix = gt_fix > 0.5         
    if not torch.any(gt_fix):
        return torch.tensor(float('nan'))

         
    pred_norm = (pred_sal - torch.mean(pred_sal,dim=-1,keepdim=True)) / (torch.std(pred_sal,dim=-1,keepdim=True) + EPS)

    return torch.sum(gt_fix * pred_norm) / (torch.sum(gt_fix) + EPS)


def cc_sphere_torch(pred_sal, gt_sal):
    """球面顶点版本的CC (PyTorch实现)"""
    pred_norm = (pred_sal - torch.mean(pred_sal,dim=-1,keepdim=True)) / (torch.std(pred_sal,dim=-1,keepdim=True) + EPS)
    gt_norm = (gt_sal - torch.mean(gt_sal,dim=-1,keepdim=True)) / (torch.std(gt_sal,dim=-1,keepdim=True) + EPS)

    y_true = gt_norm - torch.mean(gt_norm,dim=-1,keepdim=True)
    y_pred = pred_norm - torch.mean(pred_norm,dim=-1,keepdim=True)
    r1 = torch.sum(y_true * y_pred)
    r2 = torch.sqrt(torch.sum(y_pred * y_pred) * torch.sum(y_true * y_true))
    return r1 / (r2 + EPS)


def sim_sphere_torch(pred_sal, gt_sal):
    """球面顶点版本的SIM (PyTorch实现)"""
    pred = (pred_sal - torch.min(pred_sal)) / (torch.max(pred_sal) - torch.min(pred_sal) + EPS)
    gt = (gt_sal - torch.min(gt_sal)) / (torch.max(gt_sal) - torch.min(gt_sal) + EPS)

    pred = pred / (torch.sum(pred,dim=-1,keepdim=True) + EPS)
    gt = gt / (torch.sum(gt,dim=-1,keepdim=True) + EPS)
    return torch.sum(torch.minimum(pred, gt))


def kl_sphere_torch(pred_sal, gt_sal):
    """球面顶点版本的KL散度 (PyTorch实现)"""
    pred = pred_sal.float()
    gt = gt_sal.float()

    pred_prob = pred / (torch.sum(pred,dim=-1,keepdim=True) + EPS)
    gt_prob = gt / (torch.sum(gt,dim=-1,keepdim=True) + EPS)

    kl = torch.sum(gt_prob * torch.log(gt_prob / (pred_prob + EPS) + EPS))
    return kl


def compute_sphere_metrics(pred_sal, gt_sal, gt_fix,device):
    """
    参数:
        pred_sal: 预测显著性图 [L]
        gt_sal: 真实显著性图 [L]
        gt_fix: 真实注视点图 (二值) [L]

    返回:
        包含各项指标的字典
    """
    with torch.no_grad():
        return {
            'AUC': auc_judd_sphere_torch(pred_sal, gt_fix,device),
            'NSS': nss_sphere_torch(pred_sal, gt_fix),
            'CC': cc_sphere_torch(pred_sal, gt_sal),
            'SIM': sim_sphere_torch(pred_sal, gt_sal),
            'KL': kl_sphere_torch(pred_sal, gt_sal)
        }


def batch_compute_metrics(pred_sal, gt_sal, gt_fix,device):
    """
    批量计算球面指标 (支持视频序列)

    参数:
        pred_sal: [B, L] 或 [B, T, L]
        gt_sal: [B, L] 或 [B, T, L]
        gt_fix: [B, L] 或 [B, T, L]

    返回:
        包含平均指标的字典
    """
                     
    if pred_sal.dim() == 3:
        B, T, L = pred_sal.shape
        pred_sal = pred_sal.view(B * T, L)
        gt_sal = gt_sal.view(B * T, L)
        gt_fix = gt_fix.view(B * T, L)

    metrics = {'AUC': [], 'NSS': [], 'CC': [], 'SIM': [], 'KL': []}

    for i in range(pred_sal.shape[0]):
        sample_metrics = compute_sphere_metrics(
            pred_sal[i], gt_sal[i], gt_fix[i],device
        )
        for k, v in sample_metrics.items():
            if not torch.isnan(v):
                metrics[k].append(v)


    return {k: torch.mean(torch.stack(v)) if v else torch.tensor(0.0)
            for k, v in metrics.items()}