import torch
import numpy as np
import logging
import random
import pickle
import wandb
import os
from transformers import WEIGHTS_NAME, CONFIG_NAME


def set_exp_name(exp_name, args):
    exp_name += '_{}'.format(args.seed)
    return exp_name


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup(exp_name, args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    exp_path = set_exp_name(exp_name, args)
    maybe_create_dir(exp_path)

    logging.basicConfig(
        filename=os.path.join(exp_path, 'log.txt'),
        filemode='w',
        format='%(asctime)s - %(levelname)s -  %(message)s',
        datefmt='%Y-%m-%d_%H-%M-%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.wandb:
        wandb.init(settings=wandb.Settings(start_method='fork'))
        wandb_run = wandb.init(project=args.wandb_project_name,
                               entity='dagger_mgs',
                               config=args,
                               tags=args.wandb_tags,
                               name=args.wandb_run_name,
                               dir=exp_path)
        return logger, wandb_run, exp_path

    return logger, exp_path


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1000000.0


def save_model(model, save_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(save_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_path, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


def save_results(metrics, outputs, distances=None, save_path=None):
    output_path = os.path.join(save_path, 'evaluation')
    maybe_create_dir(output_path)
    pickle.dump(metrics, open(os.path.join(output_path, 'metric.pkl'), 'wb'))
    pickle.dump(outputs, open(os.path.join(output_path, 'decoding.pkl'), 'wb'))
    if distances is not None:
        pickle.dump(distances, open(os.path.join(output_path, 'plot.pkl'), 'wb'))
