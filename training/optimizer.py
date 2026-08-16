import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR

def get_optimizer_old(opt, model):
    res = {}
    sch_type = 'stagewise_80'

    base_lr = 5e-4
    max_epochs = 80

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    res['optimizer'] = optim

    if sch_type == 'fixed':
        res['epoch_lr'] = None
        res['step_lr'] = None

    elif sch_type == 'cosine':
        lr_sch = CosineAnnealingLR(optim, T_max=max_epochs, eta_min=1e-6)
        res['epoch_lr'] = lr_sch
        res['step_lr'] = None

    elif sch_type == 'warmup_cosine':
        warmup_epochs = 5
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_sch = LambdaLR(optim, lr_lambda=lr_lambda)
        res['epoch_lr'] = lr_sch
        res['step_lr'] = None

    elif sch_type == 'onecycle':
        steps_per_epoch = len(opt['train_dataloader'])
        lr_sch = OneCycleLR(
            optim,
            max_lr=base_lr * 10,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1e4
        )
        res['epoch_lr'] = None
        res['step_lr'] = lr_sch

    elif sch_type == 'stagewise_80':
        warmup_epochs = 5
        e1 = 45
        e2 = 60
        e3 = max_epochs

        lr_start = base_lr * 0.1
        lr_peak = base_lr
        lr_mid = 1e-4
        lr_refine = 4e-5
        lr_final = 1.5e-5

        def cosine_interp(lr_a, lr_b, p):
            p = min(max(p, 0.0), 1.0)
            return lr_b + 0.5 * (lr_a - lr_b) * (1.0 + math.cos(math.pi * p))

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                p = float(epoch + 1) / float(warmup_epochs)
                lr = lr_start + p * (lr_peak - lr_start)
            elif epoch < e1:
                p = float(epoch - warmup_epochs) / float(e1 - warmup_epochs)
                lr = cosine_interp(lr_peak, lr_mid, p)
            elif epoch < e2:
                p = float(epoch - e1) / float(e2 - e1)
                lr = cosine_interp(lr_mid, lr_refine, p)
            else:
                p = float(epoch - e2) / float(e3 - e2)
                lr = cosine_interp(lr_refine, lr_final, p)
            return lr / base_lr

        lr_sch = LambdaLR(optim, lr_lambda=lr_lambda)
        res['epoch_lr'] = lr_sch
        res['step_lr'] = None
    else:
        raise ValueError(f"Unknown lr_policy: {sch_type}")

    return res


def get_optimizer(opt, model):
    res = {}

    max_epochs = opt.num_epochs

    sch_type = opt.optim.sch_type
    base_lr = opt.optim.base_lr
    beta1 = opt.optim.beta1
    beta2 = opt.optim.beta2
    weight_decay = opt.optim.weight_decay

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=base_lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    res['optimizer'] = optim

    if sch_type == 'fixed':
        res['epoch_lr'] = None
        res['step_lr'] = None

    elif sch_type == 'cosine':
        lr_sch = CosineAnnealingLR(optim, T_max=max_epochs, eta_min=1e-6)
        res['epoch_lr'] = lr_sch
        res['step_lr'] = None

    elif sch_type == 'warmup_cosine':
        warmup_epochs = opt.optim.stage_epochs.warmup

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_sch = LambdaLR(optim, lr_lambda=lr_lambda)
        res['epoch_lr'] = lr_sch
        res['step_lr'] = None

    elif sch_type == 'onecycle':
        steps_per_epoch = len(opt.train_dataloader)
        lr_sch = OneCycleLR(
            optim,
            max_lr=base_lr * 10,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1e4
        )
        res['epoch_lr'] = None
        res['step_lr'] = lr_sch

    elif sch_type == 'stagewise_80':

        warmup_epochs = opt.optim.stage_epochs.warmup
        e1 = opt.optim.stage_epochs.e1
        e2 = opt.optim.stage_epochs.e2
        e3 = max_epochs

        lr_start = base_lr * opt.optim.stage_lrs.start_ratio
        lr_peak = base_lr
        lr_mid = base_lr * opt.optim.stage_lrs.mid_ratio
        lr_refine = base_lr * opt.optim.stage_lrs.refine_ratio
        lr_final = base_lr * opt.optim.stage_lrs.final_ratio

        def cosine_interp(lr_a, lr_b, p):
            p = min(max(p, 0.0), 1.0)
            return lr_b + 0.5 * (lr_a - lr_b) * (1.0 + math.cos(math.pi * p))

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                p = float(epoch + 1) / float(warmup_epochs)
                lr = lr_start + p * (lr_peak - lr_start)
            elif epoch < e1:
                p = float(epoch - warmup_epochs) / float(e1 - warmup_epochs)
                lr = cosine_interp(lr_peak, lr_mid, p)
            elif epoch < e2:
                p = float(epoch - e1) / float(e2 - e1)
                lr = cosine_interp(lr_mid, lr_refine, p)
            else:
                p = float(epoch - e2) / float(e3 - e2)
                lr = cosine_interp(lr_refine, lr_final, p)
            return lr / base_lr

        lr_sch = LambdaLR(optim, lr_lambda=lr_lambda)
        res['epoch_lr'] = lr_sch
        res['step_lr'] = None

    else:
        raise ValueError(f"Unknown lr_policy: {sch_type}")

    return res
