import os, pickle
import re
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from time import time
import copy
import torch
from loss import JointWireframeLoss
from optimizer import get_optimizer


METRIC_TAGS = {
    'Edge/Soft_Recall': 'edge/recall',
    'Edge/Soft_Precision': 'edge/precision',
    'Edge/Soft_F1': 'edge/f1',
    'Corner_Cls/Precision': 'corner_cls/precision',
    'Corner_Cls/Recall': 'corner_cls/recall',
    'Corner_Cls/F1': 'corner_cls/f1',
    'Corner/Reg_Recall_0.5m': 'corner/recall_0_5m',
    'Corner/Avg_Loc_Error_Global(m)': 'corner/error_global_m',
    'Corner/Outlier_Ratio(>0.5m)': 'corner/outlier_rate',
    'Corner/Avg_Loc_Error_TP(m)': 'corner/error_tp_m',
    'Corner/Recall_Strict_<0.3m': 'corner/recall_0_3m',
    'Corner/Recall_Loose_<0.6m': 'corner/recall_0_6m',
    'Ray_Stage1/Precision': 'ray/stage1_precision',
    'Ray_Stage1/Recall': 'ray/stage1_recall',
    'Ray_Stage1/F1': 'ray/stage1_f1',
    'Ray_Stage1/TP_Count': 'ray/stage1_tp',
    'Ray_Refine_Train/Precision': 'ray/refine_precision',
    'Ray_Refine_Train/Recall': 'ray/refine_recall',
    'Ray_Refine_Train/F1': 'ray/refine_f1',
    'Ray_Final_Cascade/Precision': 'ray/final_precision',
    'Ray_Final_Cascade/Recall': 'ray/final_recall',
    'Ray_Final_Cascade/F1': 'ray/final_f1',
    'Ray_Geom/Endpoint_Offset_L2(m)': 'ray/endpoint_offset_m',
    'Ray_Geom/Err_Angle(deg)': 'ray/angle_error_deg',
    'Ray_Geom/Err_Dist_L1(m)': 'ray/distance_error_m',
    'Ray_Geom/Err_Dist_Long_>=2.5m(m)': 'ray/distance_long_m',
    'Ray_Geom/Err_Dist_Relative(%)': 'ray/distance_relative',
    'Ray_Geom/Err_Dist_Short_<2.5m(m)': 'ray/distance_short_m',
    'Ray_Geom/Overshoot_>0.3m(%)': 'ray/overshoot',
    'Ray_Geom/Undershoot_<-0.3m(%)': 'ray/undershoot',
    'Refine_Geom/Err_Dist_Refined(m)': 'refine/distance_error_m',
    'Refine_Geom/Dist_Improvement(m)': 'refine/distance_gain_m',
    'Refine_Geom_Final/Err_Endpoint_3D_Before(m)': 'refine/endpoint_before_m',
    'Refine_Geom_Final/Err_Endpoint_3D_After(m)': 'refine/endpoint_after_m',
    'Refine_Geom_Final/Perp_Dist_Before(m)': 'refine/perp_before_m',
    'Refine_Geom_Final/Perp_Dist_After(m)': 'refine/perp_after_m',
    'Refine_Geom_Final/Dir_Angle_Before(deg)': 'refine/angle_before_deg',
    'Refine_Geom_Final/Dir_Angle_After(deg)': 'refine/angle_after_deg',
    'Refine_Geom_Final/Offset_Magnitude(m)': 'refine/offset_m',
    'Refine_Cls/Inside_Precision': 'refine/inside_precision',
    'Refine_Cls/Inside_Recall': 'refine/inside_recall',
    'Refine_Cls/Line_Precision': 'refine/line_precision',
    'Refine_Cls/Line_Recall': 'refine/line_recall',
    'Refine_Diag/Coverage_Rate(%)': 'refine/coverage',
    'Refine_Diag/Endpoint_Entropy': 'refine/endpoint_entropy',
}


def metric_tag(name):
    if name in METRIC_TAGS:
        return METRIC_TAGS[name]
    return re.sub(r'[^a-z0-9/]+', '_', str(name).lower()).strip('_')


def train_model(opt, model):

    res = get_optimizer(opt, model)
    optim, epoch_lr, step_lr = res['optimizer'], res['epoch_lr'], res['step_lr']

    model_dir = opt.log_path  
    os.makedirs(model_dir, exist_ok=True)
    summaries_dir = os.path.join(model_dir, 'summaries')  
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')  
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_dataloader = opt['train_dataloader']  
    val_dataloader = opt['val_dataloader']  
    criterion = JointWireframeLoss(opt)

    total_steps = 0
    total_val_steps = 0
    start_epoch = opt.get('start_epoch', 0)

    if opt.resume:

        ckpt_path = os.path.join(opt.resume_dir, 'checkpoints', 'model_latest.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cuda')

        start_epoch = int(ckpt['epoch']) + 1
        total_steps = int(ckpt['steps_train'])
        total_val_steps = int(ckpt['steps_val_steps'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])

        pretrained_state = ckpt['model_state_dict']
        model_state = model.state_dict()

        matched, skipped = 0, 0
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v  
                matched += 1
            else:
                skipped += 1

        model.load_state_dict(model_state)
        print(f"Resumed training: loaded={matched}, skipped={skipped}.")

        writer = SummaryWriter(summaries_dir)
        val_writer = SummaryWriter(os.path.join(summaries_dir, 'val_only'))
    else:
        writer = SummaryWriter(summaries_dir)
        val_writer = SummaryWriter(os.path.join(summaries_dir, 'val_only'))

    ema_model = ModelEMA(model, decay=0.999)

    if getattr(opt, 'resume', False) and 'ema_state_dict' in ckpt:
        ema_model.module.load_state_dict(ckpt['ema_state_dict'])
        print("Loaded EMA state from checkpoint.")

    try:

        epoch_pbar = tqdm(range(start_epoch, opt.num_epochs), desc="Training", dynamic_ncols=True)
        for epoch in epoch_pbar:

            epoch_pbar.set_description(f"Epoch {epoch}/{opt.num_epochs - 1}")

            model.train()

            if not epoch % opt.ckpt_epoch and epoch:
                save_path = os.path.join(checkpoints_dir, f'model_epoch_{epoch:04d}.pth')
                torch.save({
                    'epoch': epoch,
                    'steps_train': total_steps,
                    'steps_val_steps': total_val_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                }, save_path)

                latest = os.path.join(checkpoints_dir, 'model_latest.pth')
                try:
                    if os.path.exists(latest):
                        os.remove(latest)
                    os.link(save_path, latest)  
                except Exception:
                    import shutil
                    shutil.copyfile(save_path, latest)

            train_monitor = {}
            train_loss_log = {}
            total_train_loss_list = []
            batch_count = 0

            batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                              leave=False, dynamic_ncols=True, smoothing=0,
                              position=1, mininterval=1.0)

            for data in batch_pbar:

                optim.zero_grad(set_to_none=True)
                if total_steps % 100 == 0:
                    writer.flush()

                model_input, gt, info = data

                num_corners = gt['corner_xyz'].shape[0]
                if num_corners > 512:
                    print(
                        f"Skipping {info.get('name', 'unknown')}: "
                        f"{num_corners} corners exceeds the 512-corner limit."
                    )
                    continue

                model_input = {key: val.cuda(non_blocking=True) for key, val in model_input.items()}
                gt = {key: val.cuda(non_blocking=True) for key, val in gt.items()}
                model_input['info'] = info
                gt['info'] = info

                model_output = model(model_input)
                train_loss, raw_loss_dict = criterion(model_output, gt, epoch, opt.num_epochs)

                total_train_loss_list.append(train_loss.item())
                writer.add_scalar("loss/train_step", train_loss.item(), total_steps)

                if len(train_loss_log) == 0:
                    for key, val in raw_loss_dict.items():
                        train_loss_log[key] = [val.item()]
                else:
                    for key, val in raw_loss_dict.items():
                        train_loss_log[key].append(val.item())

                train_loss.backward()
                optim.step()
                ema_model.update(model)

                monitor_values = model.forward_moni(model_output, gt)

                if len(train_monitor) == 0:
                    for key, val in monitor_values.items():
                        train_monitor[key] = [val]
                else:
                    for key, val in monitor_values.items():
                        train_monitor[key].append(val)

                if total_steps % 100 == 0:
                    total_grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.detach().norm(2).item() ** 2
                    writer.add_scalar("optimizer/grad_norm", total_grad_norm ** 0.5, total_steps)

                    total_weight_norm = 0.0
                    for p in model.parameters():
                        total_weight_norm += p.detach().norm(2).item() ** 2
                    writer.add_scalar("optimizer/weight_norm", total_weight_norm ** 0.5, total_steps)

                current_lr = optim.param_groups[0]['lr']
                writer.add_scalar("optimizer/lr", current_lr, total_steps)

                if not total_steps % opt.steps_summary:
                    message = (
                        f"epoch={epoch} step={total_steps} "
                        f"loss={train_loss:.4f} lr={current_lr:.3e}"
                    )
                    tqdm.write(message)

                if step_lr is not None:
                    step_lr.step()

                total_steps += 1
                batch_count += 1

            for key, val_list in train_loss_log.items():
                writer.add_scalar(f"loss/train/{key}", np.mean(val_list), epoch)

            writer.add_scalar("loss/train/total", np.mean(total_train_loss_list), epoch)

            for key, val_list in train_monitor.items():
                writer.add_scalar(f"metrics/train/{metric_tag(key)}", np.mean(val_list), epoch)

            vals_edge = {}
            val_loss_log = {}
            total_val_loss_list = []  

            with torch.no_grad():
                t1 = time()
                ema_model.module.eval()  

                for data in val_dataloader:
                    model_input, gt, info = data

                    num_corners = gt['corner_xyz'].shape[0]
                    if num_corners > 512:
                        continue

                    model_input = {key: val.cuda() for key, val in model_input.items()}
                    gt = {key: val.cuda() for key, val in gt.items()}
                    model_input['info'] = info

                    model_output = ema_model.module(model_input, mode='train')

                    val_loss, val_raw_loss_dict = criterion(model_output, gt, epoch, opt.num_epochs)

                    total_val_loss_list.append(val_loss.item())
                    writer.add_scalar("loss/val_step", val_loss.item(), total_val_steps)

                    if len(val_loss_log) == 0:
                        for key, val in val_raw_loss_dict.items():
                            val_loss_log[key] = [val.item()]
                    else:
                        for key, val in val_raw_loss_dict.items():
                            val_loss_log[key].append(val.item())

                    val_edge_dict = ema_model.module.forward_moni(model_output, gt)

                    if len(vals_edge) == 0:
                        for key, val in val_edge_dict.items():
                            vals_edge[key] = [val]
                    else:
                        for key, val in val_edge_dict.items():
                            vals_edge[key].append(val)

                    for key, value in val_edge_dict.items():
                        writer.add_scalar(
                            f"metrics/val_step/{metric_tag(key)}", value, total_val_steps
                        )

                    total_val_steps += 1

                val_mean_total = np.mean(total_val_loss_list)
                writer.add_scalar("loss/val/total", val_mean_total, epoch)

                train_mean_total = np.mean(total_train_loss_list)
                writer.add_scalar("loss/total", train_mean_total, epoch)
                val_writer.add_scalar("loss/total", val_mean_total, epoch)

                for key, val_list in val_loss_log.items():
                    writer.add_scalar(f"loss/val/{key}", np.mean(val_list), epoch)

                for key in val_loss_log.keys():
                    if key in train_loss_log:
                        train_mean = np.mean(train_loss_log[key])
                        val_mean = np.mean(val_loss_log[key])

                        writer.add_scalar(f"loss/{key}", train_mean, epoch)
                        val_writer.add_scalar(f"loss/{key}", val_mean, epoch)

                for key, val_list in vals_edge.items():
                    writer.add_scalar(
                        f"metrics/val/{metric_tag(key)}", np.mean(val_list), epoch
                    )

                print(
                    f"Epoch {epoch}: train_loss={train_mean_total:.4f}, "
                    f"val_loss={val_mean_total:.4f}, val_time={time() - t1:.1f}s."
                )
                if epoch == opt.num_epochs - 1:
                    val_path = os.path.join(model_dir, 'final_val.pkl')
                    with open(val_path, 'wb') as f:
                        pickle.dump(vals_edge, f)

                model.train()  

            if epoch_lr is not None:
                epoch_lr.step()

            torch.save({
                'epoch': epoch,
                'steps_train': total_steps,
                'steps_val_steps': total_val_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'ema_state_dict': ema_model.module.state_dict(),  
                'scheduler_state_dict': epoch_lr.state_dict(),
            }, os.path.join(checkpoints_dir, f'model_final.pth'))

    finally:
        writer.close()
        val_writer.close()


class ModelEMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay

        self.module = copy.deepcopy(model)
        self.module.eval()

        for param in self.module.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        ema_state_dict = self.module.state_dict()
        model_state_dict = model.state_dict()

        for key in ema_state_dict:
            ema_param = ema_state_dict[key]
            model_param = model_state_dict[key]

            if ema_param.dtype.is_floating_point:
                ema_param.copy_(ema_param * self.decay + (1. - self.decay) * model_param)
            else:
                ema_param.copy_(model_param)
