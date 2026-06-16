# <editor-fold desc="header">
import datetime
import logging
import os
import os.path as op

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import cfg


# </editor-fold>


class BaseTrainer:
    def __init__(self, rank, world_size):
        super(BaseTrainer, self).__init__()
        # fix
        self.rank = rank
        self.world_size = world_size
        if self.rank == 0:
            now_time = datetime.datetime.now()
            now_time = 'Log_v%02d%02d%02d%02d/' % (now_time.month, now_time.day, now_time.hour, now_time.minute)
            exp_dir = op.join(cfg.exp_out_dir, now_time)
            tb_log = op.join(exp_dir, 'tb_log')
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(tb_log, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_log)
            self.ckpt_dir = op.join(exp_dir, 'ckpt')
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def compute_f1(self, logit, y, is_pred=False):
        with torch.no_grad():
            if is_pred:
                pred = logit.squeeze(1)
            else:
                if len(logit.shape) == 3:
                    pred = F.sigmoid(logit) > 0.5
                elif len(logit.shape) == 4:
                    pred = logit.argmax(1)  # ori [b,h,w]
            y_ = y.squeeze(1)
            matched = (pred * y_).sum((1, 2))
            pred_sum = pred.sum((1, 2))
            y_sum = y_.sum((1, 2))
            p = matched / (pred_sum + 1e-8)
            r = matched / (y_sum + 1e-8)
            f1 = (2 * p * r / (p + r + 1e-8)).mean().item()
        return f1, p.mean().item(), r.mean().item()

    def write_log(self, cnt, losses_record):
        if self.rank == 0:
            for loss_name, loss_value in losses_record.items():
                self.tb_writer.add_scalar('losses/{}'.format(loss_name.strip()), loss_value.val, global_step=cnt)

    def print_log(self, step, losses_record):
        if self.rank != 0:
            return
        lr = self.optimizer.param_groups[0]['lr']
        output = 'Step: %6d; lr:%.2e;' % (step, lr)
        for name, loss in losses_record.items():
            output += ' %s: %5.4f;' % (name, loss.val)
        logging.info(output)

    def load_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            if not op.isfile(ckpt_path):
                logging.warning(f'Checkpoint not found, skipping load: {ckpt_path}')
                return
            self.ckpt = torch.load(cfg.ckpt, map_location='cpu')
            miss, unexpect = self.model.load_state_dict(self.modify_cp_dict(self.ckpt['model']), strict=False)
            print(f"Loaded model from {cfg.ckpt}. Missed keys: {miss}, Unexpected keys: {unexpect}")

    def modify_cp_dict(self, cp_dict):
        new_cp_dict = {}
        for key in cp_dict:
            new_key = key.replace('module.', '')
            new_cp_dict[new_key] = cp_dict[key]
        return new_cp_dict

    def save_ckpt(self, step, score):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state_dict, op.join(self.ckpt_dir, 'Step%s_Score%5.4f.pth' % (step, score)))
