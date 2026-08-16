import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import torch
import copy

from loss import JointWireframeLoss
from optimizer import get_optimizer

def train_model(opt, model):
    optim = get_optimizer(opt, model)
    optim, epoch_lr, step_lr = optim['optimizer'], optim['epoch_lr'], optim['step_lr']

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

    ema_model = ModelEMA(model, decay=0.999)

    if getattr(opt, 'resume', False):
        ckpt_path = os.path.join(opt.resume_dir, 'checkpoints', 'model_latest.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cuda')

        start_epoch = int(ckpt['epoch']) + 1
        total_steps = int(ckpt.get('steps_train', 0))
        total_val_steps = int(ckpt.get('steps_val_steps', 0))

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

        epoch_lr.load_state_dict(ckpt['scheduler_state_dict'])

        ema_model.module.load_state_dict(ckpt['ema_state_dict'])

    writer = SummaryWriter(summaries_dir)
    val_writer = SummaryWriter(os.path.join(summaries_dir, 'val_only'))

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
                    'ema_state_dict': ema_model.module.state_dict(),
                    'scheduler_state_dict': epoch_lr.state_dict() if epoch_lr else None,
                }, save_path)

                latest = os.path.join(checkpoints_dir, 'model_latest.pth')
                try:
                    if os.path.exists(latest):
                        os.remove(latest)
                    os.link(save_path, latest)
                except Exception:
                    import shutil
                    shutil.copyfile(save_path, latest)

            train_loss_log = {}
            total_train_loss_list = []

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
                    continue

                model_input = {key: val.cuda(non_blocking=True) for key, val in model_input.items()}
                gt = {key: val.cuda(non_blocking=True) for key, val in gt.items()}

                model_output = model(model_input)
                train_loss, raw_loss_dict = criterion(model_output, gt, epoch, opt.num_epochs)

                train_loss.backward()
                optim.step()
                ema_model.update(model)

                total_train_loss_list.append(train_loss.item())
                writer.add_scalar("loss/train_step", train_loss.item(), total_steps)
                writer.add_scalar("optimizer/lr", optim.param_groups[0]['lr'], total_steps)

                if len(train_loss_log) == 0:
                    for key, val in raw_loss_dict.items():
                        train_loss_log[key] = [val.item()]
                else:
                    for key, val in raw_loss_dict.items():
                        train_loss_log[key].append(val.item())

                if not total_steps % opt.steps_summary:
                    current_lr = optim.param_groups[0]['lr']
                    message = (
                        f"epoch={epoch} step={total_steps} "
                        f"loss={train_loss:.4f} lr={current_lr:.3e}"
                    )
                    tqdm.write(message)

                if step_lr is not None:
                    step_lr.step()

                total_steps += 1

            for key, val_list in train_loss_log.items():
                writer.add_scalar(f"loss/train/{key}", np.mean(val_list), epoch)
            writer.add_scalar("loss/train/total", np.mean(total_train_loss_list), epoch)

            val_loss_log = {}
            total_val_loss_list = []

            with torch.no_grad():
                ema_model.module.eval()

                for data in val_dataloader:
                    model_input, gt, info = data
                    if gt['corner_xyz'].shape[0] > 512:
                        continue

                    model_input = {key: val.cuda(non_blocking=True) for key, val in model_input.items()}
                    gt = {key: val.cuda(non_blocking=True) for key, val in gt.items()}

                    model_output = ema_model.module(model_input, mode='train')
                    val_loss, val_raw_loss_dict = criterion(model_output, gt, epoch, opt.num_epochs)

                    total_val_loss_list.append(val_loss.item())
                    if len(val_loss_log) == 0:
                        for key, val in val_raw_loss_dict.items():
                            val_loss_log[key] = [val.item()]
                    else:
                        for key, val in val_raw_loss_dict.items():
                            val_loss_log[key].append(val.item())

                    total_val_steps += 1

            val_mean_total = np.mean(total_val_loss_list)
            train_mean_total = np.mean(total_train_loss_list)

            writer.add_scalar("loss/val/total", val_mean_total, epoch)
            writer.add_scalar("loss/total", train_mean_total, epoch)
            val_writer.add_scalar("loss/total", val_mean_total, epoch)

            for key, val_list in val_loss_log.items():
                writer.add_scalar(f"loss/val/{key}", np.mean(val_list), epoch)

            for key in val_loss_log.keys():
                if key in train_loss_log:
                    writer.add_scalar(f"loss/{key}", np.mean(train_loss_log[key]), epoch)
                    val_writer.add_scalar(f"loss/{key}", np.mean(val_loss_log[key]), epoch)

            print(f"Epoch {epoch}: train_loss={train_mean_total:.4f}, val_loss={val_mean_total:.4f}.")

            if epoch_lr is not None:
                epoch_lr.step()

        torch.save({
            'epoch': opt.num_epochs - 1,
            'steps_train': total_steps,
            'steps_val_steps': total_val_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'ema_state_dict': ema_model.module.state_dict(),
            'scheduler_state_dict': epoch_lr.state_dict() if epoch_lr else None,
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
