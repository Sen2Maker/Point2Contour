import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
import loss
from train_debug import train_model as train_debug
from train import train_model as train_n


def config_loss(loss_schedule):
    loss_handler = loss.LossHandlerPoint()
    loss_handler.parse_config(loss_schedule)
    return loss_handler


def eval_loss(output, gt, loss_schedule):
    res = {}
    for name, config in loss_schedule.items():
        if config.enable:
            loss_term = getattr(loss, name)(output, gt, config)

            if isinstance(loss_term, dict):
                res.update(loss_term)
            else:
                res[name] = loss_term

    return res


def eval_val_loss(output, gt, output_type, metric='L1'):
    return getattr(loss, 'val_loss')(output, gt, output_type, metric)
