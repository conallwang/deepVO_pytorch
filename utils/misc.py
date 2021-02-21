import torch
import os
import shutil

from torch.autograd import Variable

def pre_create_file_train(model_dir, log_dir, args):
    models_path = f'{model_dir}/{args.net_name}'
    logs_path = f'{log_dir}/{args.net_name}'
    model_path = f'{models_path}/{args.dir0}'
    log_path = f'{logs_path}/{args.dir0}'

    if not os.path.exists(models_path):
        os.mkdir(models_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.mkdir(log_path)

    return model_path, log_path

def to_var(x):
    if torch.cuda.is_available():
        return Variable(x).cuda()
    else:
        return Variable(x)

def display_loss_tb(hour_per_epoch, pbar, step, step_per_epoch, optimizer, loss, writer, step_global):
    state_msg = (
        f'{hour_per_epoch:.3f} [{step:03}/{step_per_epoch:03}] lr {optimizer.param_groups[0]["lr"]:.7f}: {loss}'
    )
    pbar.set_description(state_msg)
    writer.add_scalars('./train-val', {'loss_t': loss}, step_global)