from collections import defaultdict, OrderedDict
from copy import deepcopy
from pprint import pformat
from scipy.stats import kendalltau
from timeit import default_timer as timer
from enum import Enum

import hashlib
import logging
import math
import numpy as np
import os
import random
import shelve
import torch
from tqdm import tqdm

import torch.distributed as dist

import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import Adadelta, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import IterableDataset, DataLoader, Dataset

import pickle
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


import seq_level.gpt2.guided.score_network as score_network_utils
import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils
timer_context = ggs_utils.TimerContext()
def _hash_tensor(obj):
    return hashlib.sha1(pickle.dumps(obj)).hexdigest()


def _hash_model(model):
    return hashlib.sha1(pickle.dumps(model)).hexdigest()

MODEL_ID = None

torch.multiprocessing.set_sharing_strategy('file_system')
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

class InstanceType(Enum):
    PERTURBED=1
    NON_PERTURBED=2

def multigpu_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_dataloader(args, dataset, rank=0, world_size=1, batch_size=1, collate_fn=None):
    if args.multigpu:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader

def cleanup():
    dist.destroy_process_group()
class RingBuffer:
    def __init__(self, max_size=1000, persistence='none', 
                  persistent_file_path=None, shuffle=True, 
                  iter_device=None, on_device=False):

        self.max_size = max_size
        self.persistence = persistence
        
        self.train_queue = []
        self.valid_queue = []

        self.valid_idxs = set([])
        self.train_idxs = set([])

        if persistence == 'none':
            self.db = {}

        elif persistence == 'shelve':
            self.db = shelve.open(persistent_file_path)

        self.db_counter = {}

        self.shuffle = shuffle
        self.iter_device = None

        self.on_device = on_device

    def __len__(self):
        return len(self.train_queue)

    def append(self, idx, type, batch_id, batch, model, sequences, 
                distances, log_rho=None, noise_mag=None, 
                rng_state=None, perturb_type=None):

        if idx not in self.valid_idxs and idx not in self.train_idxs:
            if random.random() > 0.9:
                self.valid_idxs.add(idx)
            else:
                self.train_idxs.add(idx)
        
        # Randomly w/ prob 0.9, add to train queue, and
        # with 0.1 add to valid queue. 
        queue = self.train_queue
        queue_max_size = self.max_size
        if idx in self.valid_idxs:
            queue = self.valid_queue
            queue_max_size = int(0.2 * self.max_size)

        print(f"Id: {idx:>20}:: " +
              f"Instances: {len(self.train_idxs):>4}/{len(self.valid_idxs):<4}, " + 
              f" Queue Sizes: {len(self.train_queue):>4}/{len(self.valid_queue):<4}," +
              f" DB size: {len(self.db):>6}", end='\r')
        if len(queue) >= queue_max_size:
            (_, _, old_batch_key, old_model_key,
             old_seq_key, _, _, _, _, _) = queue.pop(0)
            logging.debug("Removing item from Queue: " +
                          f"Batch: {old_batch_key} " +
                          f"Model: {old_model_key}.")

            self.db_counter[old_model_key] -= 1
            if self.db_counter[old_model_key] == 0:
                del self.db_counter[old_model_key]
                del self.db[old_model_key]

            self.db_counter[old_batch_key] -= 1
            if self.db_counter[old_batch_key] == 0:
                del self.db_counter[old_batch_key]
                del self.db[old_batch_key]

            self.db_counter[old_seq_key] -= 1
            if self.db_counter[old_seq_key] == 0:
                del self.db_counter[old_seq_key]
                del self.db[old_seq_key]

        # batch_key = f"batch_{_hash_tensor(batch)}"
        batch_key = f"batch_{batch_id}"
        if batch_key not in self.db:
            if not self.on_device:
                batch = deepcopy(batch).to(device=torch.device("cpu"))
            self.db[batch_key] = batch
            self.db_counter[batch_key] = 0
        self.db_counter[batch_key] += 1

        # global MODEL_ID
        model_key = f"model_{MODEL_ID}"
        # model_key = f"model_{_hash_model(model)}"

        if model_key not in self.db:
            if not self.on_device:
                model = deepcopy(model).to(device=torch.device("cpu"))
            self.db[model_key] = model
            self.db_counter[model_key] = 0
        self.db_counter[model_key] += 1

        sequences_key =  f"seq_{batch_id}_{MODEL_ID}_{rng_state}"
        if sequences_key not in self.db:
            if not self.on_device:
                sequences = deepcopy(sequences).to(device=torch.device("cpu"))
            self.db[sequences_key] = sequences
            self.db_counter[sequences_key] = 0
        self.db_counter[sequences_key] += 1

        if not self.on_device:
            distances = distances.cpu()

        queue.append((idx, type, batch_key, model_key, 
                        sequences_key, distances, rng_state, 
                        log_rho, noise_mag, perturb_type))

    def get_iterators(self, args, shuffle=True):
        def _batch_generator(iterable, shuffle=True):
            if shuffle:
                iterable = random.sample(iterable, len(iterable))

            for (idx, type, batch_key, model_key, sequences_key, 
                    distances, rng_state, log_rho, noise_mag, perturb_type) in iterable:

                batch = self.db[batch_key]
                sequences = self.db[sequences_key]
                model = self.db[model_key]

                if type == InstanceType.PERTURBED:
                    model_ = deepcopy(model)
                    inp, target = batch[:, :-1], batch[:, 1:]
                    model_.eval()
                    model_with_grad, _ = ggs_utils.mle_grad(model_, inp, target, 
                                            args.pad_token_id, args.max_grad_norm)
                    model, log_rho_, noise_mag_, rng_ = ggs_utils.perturb_single(model, 
                            model_with_grad, args.ggs_noise, noise_scale=args.noise_scale,
                                        perturb_type=perturb_type, rng_state=rng_state)
                
                if distances.size(0) != batch.size(0):
                    logging.error(
                        f"Distance: {distances.size(0)}, Batch: ({batch.size()}), {batch_key} Sequence: {sequences_key}" + \
                        f"Model: {model_key}.")
                    continue
                yield (idx, type, batch, model, sequences, 
                        distances, rng_state, perturb_type, log_rho, noise_mag, model_with_grad)

        return (_batch_generator(self.train_queue, shuffle=shuffle), 
                    _batch_generator(self.valid_queue, shuffle=False))

    def _generate_model_info(self, iterable, min_context_len):
        idx2model_info = OrderedDict()
        for (idx, type, batch_key, model_key, sequences_key, 
                distances, rng_state, log_rho, noise_mag, perturb_type) in iterable:

            batch_size = self.db[batch_key].size(1)
            if batch_size < min_context_len:
                continue

            if idx not in idx2model_info:
                idx2model_info[idx] = {
                    'non_pert': None,
                    'pert': [],
                    'batch_key': batch_key,
                    'model_key': model_key,
                }
            assert idx2model_info[idx]['batch_key'] == batch_key
            assert idx2model_info[idx]['model_key'] == model_key

            entry = (sequences_key, distances, rng_state, log_rho, noise_mag, perturb_type)
            if type == InstanceType.NON_PERTURBED:
                idx2model_info[idx]['non_pert'] = entry
            else:
                idx2model_info[idx]['pert'].append(entry)
        return idx2model_info

    def _get_model_at_idx(self, idx2model_info, args, idx, device=0):
        data = idx2model_info[idx]

        idx2model = {}
        batch = self.db[data['batch_key']]
        model = self.db[data['model_key']]

        idx2model['idx'] = idx
        idx2model['batch'] = batch
        idx2model['model'] = model

        batch = batch.to(device)
        model = model.to(device)
        inp, target = batch[:, :-1], batch[:, 1:]

        sequences_key_np, distances_np, _, _, _, _ = data['non_pert']


        sequence_np = self.db[sequences_key_np]
        idx2model['non_pert'] = (sequence_np, distances_np)
        sequence_np = sequence_np.to(device)
        distances_np = distances_np.to(device)

        idx2model['pert'] = []
        model_ = deepcopy(model)

        model_.eval()
        with torch.set_grad_enabled(True):
            model_with_grad, _ = ggs_utils.mle_grad(model_, inp, target, 
                                    args.pad_token_id, args.max_grad_norm)

        for (sequences_key_p, distances_p, rng_state, 
                    log_rho, noise_mag,  perturb_type) in data['pert']:
            sequence_p = self.db[sequences_key_p]
            sequence_p = sequence_p.to(device)
            distances_p = distances_p.to(device)
            if args.zero_dist_only and \
                not perturb_type != ggs_utils.PerturbationType.ONLY_NOISE:
                    continue

            if args.mle_dist_only and \
                not perturb_type != ggs_utils.PerturbationType.MLE_GRAD_W_NOISE:
                    continue

            model_p, log_rho_, noise_mag_, rng_ = ggs_utils.perturb_single(model, 
                    model_with_grad, args.ggs_noise, noise_scale=args.noise_scale,
                                perturb_type=perturb_type, rng_state=rng_state)

            entry = (model_p, sequence_p, distances_p, log_rho_, noise_mag_)
            idx2model['pert'].append(entry)
        return idx2model

    def _batch_generator_v2(self, iterable, args, shuffle=True):
        min_context_len = args.context_length + 2
        idx2model_info = self._generate_model_info(iterable, min_context_len)
        items = idx2model_info.keys()
        if shuffle:
            items = random.sample(items, len(idx2model_info))

        idx2model = {}
        for idx in items:
            idx2model = self._get_model_at_idx(idx2model_info, args, idx)
            yield (idx, idx2model)

    def get_iterators_v2(self, args, shuffle=True):
        return (self._batch_generator_v2(self.train_queue, args, shuffle=True), 
                 self._batch_generator_v2(self.valid_queue, args, shuffle=False))

    def get_dataloaders(self, args, device, world_size, shuffle=True):
        train_dataset = ScoringNetworkTrainingDataset(self, 'train', args, shuffle=shuffle, device=device)
        valid_dataset = ScoringNetworkTrainingDataset(self, 'valid', args, device=device)
        train_dataloader = get_dataloader(args, train_dataset, device, world_size, collate_fn=lambda x: x[0])
        valid_dataloader = get_dataloader(args, valid_dataset, device, world_size, collate_fn=lambda x: x[0])
        return (train_dataloader, valid_dataloader)

class ScoringNetworkIterableTrainingDataset(IterableDataset):
    def __init__(self, buffer, split, args, shuffle=False, device=0):
        self.args = args
        if split == 'train':
            iterator = buffer.train_queue
        elif split == 'valid':
            iterator = buffer.valid_queue
        else:
            raise ValueError(f"Illegal Split: {split}")
        self.device = device

        self.iterator = iterator
        self.buffer = buffer
        self.idx2model_info = buffer._generate_model_info(iterator, 
                                min_context_len=args.context_length + 2)
        self.shuffle = shuffle

    def __iter__(self):
        items = self.idx2model_info.keys()
        if self.shuffle:
            items = random.sample(items, len(self.idx2model_info))

        for idx in self.idx2model_info.keys():
            yield self.buffer._get_model_at_idx(self.idx2model_info, self.args, idx, self.device)

class ScoringNetworkTrainingDataset(Dataset):
    def __init__(self, buffer, split, args, shuffle=False, device=0):
        self.args = args
        if split == 'train':
            iterator = buffer.train_queue
        elif split == 'valid':
            iterator = buffer.valid_queue
        else:
            raise ValueError(f"Illegal Split: {split}")

        self.iterator = iterator
        self.buffer = buffer
        self.idx2model_info = buffer._generate_model_info(iterator, 
                                min_context_len=args.context_length + 2)
        self.idx2model_keys = [x for x in self.idx2model_info.keys()]
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return len(self.idx2model_keys)

    def __getitem__(self, idx):
        data_key = self.idx2model_keys[idx]
        return self.buffer._get_model_at_idx(self.idx2model_info, self.args, data_key, self.device)

class ScorerAnalysis:
    def __init__(self, prefix):
        # idx => {'original': (true_dist, pred_dist), 
        #         'perturbed': [(true_dist, pred_dist)]}
        self.scorer_diff_fit_dict = {}
        self.cuml_loss = 0.
        self.cuml_pert_loss = 0.
        self.cuml_non_pert_loss = 0.

        self.cuml_scorer_fit_all = 0.
        self.cuml_scorer_fit_pert = 0.
        self.cuml_scorer_fit_non_pert = 0.

        self.num_docs = 1
        self.num_docs_pert = 1
        self.num_docs_non_pert = 1

        self.loss_list = []
        self.true_dist = []
        self.pred_dist = []

        self.true_dist_pert = []
        self.true_dist_non_pert = []
        self.pred_dist_pert = []
        self.pred_dist_non_pert = []
        self.loss_list_pert = []
        self.loss_list_non_pert = []

        self.disagreements = 0.
        self.mean_disagreement = 0.
        self.disagreement_count = 1
        
        self.cuml_pred_dist = 0.
        self.cuml_true_dist = 0.
        
        self.cuml_pred_dist_pert = 0.
        self.cuml_pred_dist_non_pert = 0.
        self.cuml_true_dist_pert = 0.
        self.cuml_true_dist_non_pert = 0.

        self.prefix = prefix

    def __call__(self, idx, type, true_dist, pred_dist):
        losses = (((true_dist - pred_dist)**2)**0.5)
        batch_size = losses.size(0)
        self.cuml_loss += losses.sum().item()

        self.true_dist += true_dist.tolist()
        self.pred_dist += pred_dist.tolist()
        self.loss_list += losses.tolist()

        scorer_fit = (torch.abs((true_dist - pred_dist)/true_dist))\
                        .sum().item()
        
        self.cuml_scorer_fit_all += scorer_fit
        self.num_docs += batch_size

        if idx not in self.scorer_diff_fit_dict:
            self.scorer_diff_fit_dict[idx] = {
                                   'non_pert': None, 
                                   'perturbed': [],
                                }
        
        self.cuml_pred_dist += pred_dist.sum().item()
        self.cuml_true_dist += true_dist.sum().item()
        if type == InstanceType.PERTURBED:
            self.num_docs_pert += batch_size
            self.cuml_pert_loss += losses.sum().item()
            self.true_dist_pert += true_dist.tolist()
            self.pred_dist_pert += pred_dist.tolist()
            self.loss_list_pert += losses.tolist()
            self.cuml_scorer_fit_pert += scorer_fit
            self.scorer_diff_fit_dict[idx]['perturbed'].append((true_dist.sum(), 
                                                                pred_dist.sum()))
            self.cuml_pred_dist_pert += pred_dist.sum().item()
            self.cuml_true_dist_pert += true_dist.sum().item()
        else:
            self.num_docs_non_pert += batch_size
            self.cuml_non_pert_loss += losses.sum().item()
            self.true_dist_non_pert += true_dist.tolist()
            self.pred_dist_non_pert += pred_dist.tolist()
            self.loss_list_non_pert += losses.tolist()
            self.cuml_scorer_fit_non_pert += scorer_fit
            self.scorer_diff_fit_dict[idx]['non_pert'] = (true_dist.sum(), 
                                                          pred_dist.sum())
            self.cuml_pred_dist_non_pert += pred_dist.sum().item()
            self.cuml_true_dist_non_pert += true_dist.sum().item()

    def reset(self):
        self.scorer_diff_fit_dict = {}
        self.cuml_pert_loss = 0.
        self.cuml_non_pert_loss = 0.

        self.cuml_scorer_fit_all = 0.
        self.cuml_scorer_fit_pert = 0.
        self.cuml_scorer_fit_non_pert = 0.

        self.num_docs_pert = 1
        self.num_docs_non_pert = 1

        self.loss_list = []
        self.true_dist = []
        self.pred_dist = []

        self.true_dist_pert = []
        self.true_dist_non_pert = []
        self.pred_dist_pert = []
        self.pred_dist_non_pert = []
        self.loss_list_pert = []
        self.loss_list_non_pert = []

        self.cuml_pred_dist = 0.
        self.cuml_true_dist = 0.
        
        self.cuml_pred_dist_pert = 0.
        self.cuml_pred_dist_non_pert = 0.
        self.cuml_true_dist_pert = 0.
        self.cuml_true_dist_non_pert = 0.

    def get_metrics(self):
        for idx, diff_dict in self.scorer_diff_fit_dict.items():
            non_pert_true_score, non_pert_pred_score = diff_dict['non_pert']

            pertured_scores = diff_dict['perturbed']
            for perturb_true_score, perturb_pred_score in pertured_scores:
                if (perturb_true_score - non_pert_true_score) == 0:
                    logging.debug(f"For idx: {idx}," + 
                        f" perturbed score: {perturb_true_score:.3f}" + 
                        f" non_pert score: {non_pert_true_score:.3f}" + 
                         " are equal.")
                    continue

                pred_diff = perturb_pred_score - non_pert_pred_score
                true_diff = perturb_true_score - non_pert_true_score
                self.disagreements += (pred_diff/true_diff < 0).item()
                self.mean_disagreement += torch.abs((pred_diff - true_diff)/true_diff).item()
                self.disagreement_count += 1

        bins = np.array([0, 0.125, 0.25, 0.5, 1, 2, 3, 4] + list(range(10, 110, 10)))
        idxs = np.digitize(np.array(self.loss_list), bins)
        loss_digitized = bins[idxs - 1]

        idxs = np.digitize(np.array(self.loss_list_pert), bins)
        loss_pert_digitized =  bins[idxs - 1]

        idxs = np.digitize(np.array(self.loss_list_non_pert), bins)
        loss_non_pert_digitized =  bins[idxs - 1]

        return {
          f"{self.prefix}/corr_all": kendalltau(self.true_dist, self.pred_dist)[0],
          f"{self.prefix}/corr_pert": kendalltau(self.true_dist_pert, self.pred_dist_pert)[0],
          f"{self.prefix}/corr_non_pert": kendalltau(self.true_dist_non_pert,
                                                     self.pred_dist_non_pert)[0],

          f"{self.prefix}/loss_mean": self.cuml_loss / self.num_docs,
          f"{self.prefix}/loss_mean_pert": self.cuml_pert_loss / self.num_docs_pert,
          f"{self.prefix}/loss_mean_non_pert": self.cuml_non_pert_loss / self.num_docs_non_pert, 

          f"{self.prefix}/loss_u1": np.mean(loss_digitized < 1) ,
          f"{self.prefix}/loss_pert_u1": np.mean(loss_pert_digitized < 1),
          f"{self.prefix}/loss_non_pert_u1": np.mean(loss_non_pert_digitized < 1),

          f"{self.prefix}/loss_u10": np.mean(loss_digitized < 10) ,
          f"{self.prefix}/loss_pert_u10": np.mean(loss_pert_digitized < 10),
          f"{self.prefix}/loss_non_pert_u10": np.mean(loss_non_pert_digitized < 10), 

          f"{self.prefix}/loss_u20": np.mean(loss_digitized < 20) ,
          f"{self.prefix}/loss_pert_u20": np.mean(loss_pert_digitized < 20),
          f"{self.prefix}/loss_non_pert_u20": np.mean(loss_non_pert_digitized < 20), 

          f"{self.prefix}/loss_u30": np.mean(loss_digitized < 30) ,
          f"{self.prefix}/loss_pert_u30": np.mean(loss_pert_digitized < 30),
          f"{self.prefix}/loss_non_pert_u30": np.mean(loss_non_pert_digitized < 30), 

          f"{self.prefix}/scorer_fit_all": self.cuml_scorer_fit_all / self.num_docs,
          f"{self.prefix}/scorer_fit_pert": self.cuml_scorer_fit_pert / self.num_docs_pert,
          f"{self.prefix}/scorer_fit_non_pert": self.cuml_scorer_fit_non_pert / self.num_docs_non_pert,

          f"{self.prefix}/disagreements": self.disagreements / self.disagreement_count,
          f"{self.prefix}/mean_disagreements": self.mean_disagreement / self.disagreement_count,

          f"{self.prefix}/true_dist_mean": self.cuml_true_dist/self.num_docs, 
          f"{self.prefix}/true_dist_pert_mean": self.cuml_true_dist_pert/self.num_docs_pert, 
          f"{self.prefix}/true_dist_non_pert_mean": self.cuml_true_dist_non_pert/self.num_docs_non_pert,

          f"{self.prefix}/pred_dist_mean":self.cuml_pred_dist/self.num_docs, 
          f"{self.prefix}/pred_dist_pert_mean":self.cuml_pred_dist_pert/self.num_docs_pert,
          f"{self.prefix}/pred_dist_non_pert_mean":self.cuml_pred_dist_non_pert/self.num_docs_non_pert, 
        }

def calculate_scorer_loss(data, loss_func_type='mse', ggs_beta=1.0):
    loss = 0.
    if loss_func_type == 'mse':
        for (dist, pred_dist, _, _) in data:
            loss += F.mse_loss(
                        pred_dist,
                        dist.view(-1, 1),
                        reduction='none',)
        loss /= len(data)
    elif loss_func_type == 'mse-diff':
        non_perturb_data = data[0]
        assert InstanceType.NON_PERTURBED == non_perturb_data[-1]

        a = (non_perturb_data[0].view(-1, 1) - non_perturb_data[1])**2
        loss += a
        for perturb_data in  data[1:]:
            assert InstanceType.PERTURBED == perturb_data[-1]
            b = (perturb_data[0].view(-1, 1) - perturb_data[1])**2
            loss += b
            loss += ((perturb_data[0] - non_perturb_data[0]).view(-1, 1) - (perturb_data[1] - non_perturb_data[1]))**2
            loss /= 3
        loss /= len(data[1:])
    elif loss_func_type == 'kl' or loss_func_type == 'tv':
        def compute_weight(dist, perturb_dist, log_rhos, beta):
            ws = beta * (dist - perturb_dist).clamp(max=1e16)
            ws = ws - log_rhos.to(ws.device)
            log_ws = torch.log_softmax(ws, dim=-1)
            return log_ws

        non_perturb_data = data[0]
        assert InstanceType.NON_PERTURBED == non_perturb_data[-1]
        curr_true_dist, curr_pred_dist =  non_perturb_data[:2]
        curr_true_dist = curr_true_dist.view(-1, 1)
        curr_pred_dist = curr_pred_dist.view(-1, 1)
        perturb_true_dist = []
        perturb_pred_dist = []
        log_rhos = []
        for perturb_data in  data[1:]:
            perturb_true_dist.append(perturb_data[0])
            perturb_pred_dist.append(perturb_data[1].squeeze(-1))
            log_rhos.append(perturb_data[2].item())

        perturb_true_dist = torch.stack(perturb_true_dist, dim=-1)
        perturb_pred_dist = torch.stack(perturb_pred_dist, dim=-1)
        log_rhos = torch.tensor(log_rhos)
        true_weights = compute_weight(curr_true_dist, perturb_true_dist, log_rhos, ggs_beta)
        pred_weights = compute_weight(curr_pred_dist, perturb_pred_dist, log_rhos, ggs_beta)
        if loss_func_type == 'kl':
            loss = F.kl_div(pred_weights, true_weights, 
                            reduction='none', log_target=True) \
                        .sum(dim=-1, keepdim=True)
        elif loss_func_type == 'tv':
            loss = torch.abs(torch.exp(pred_weights) - torch.exp(true_weights)) \
                        .sum(dim=-1, keepdim=True)
    return loss


def start_scorer_training_data_accumulation(buffer, dataset, model, score_model, tokenizer, args):
    world_size = torch.cuda.device_count()
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(10000)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=accumulate_scorer_training_data_v2, 
                args=(rank % 4, dataset, model, score_model, tokenizer, args, queue, world_size))
        processes.append(p)
        p.start()



    while True:
        try:
            data = queue.get(timeout=100)
            buffer.append(*data)
            del data
        except Exception:
            break
    
    for p in processes:
        p.join()


def accumulate_scorer_training_data_v2(device, dataset, model, score_model, tokenizer, args, queue, world_size=1):
    if args.multigpu:
        multigpu_setup(device, world_size)
    
    train_dataloader = get_dataloader(args, dataset, device, world_size, batch_size=1)
    train_dataloader.sampler.set_epoch(0)

    model = model.to(device)
    score_model = score_model.to(device)
    # model = DDP(model, device_ids=[device], 
                # output_device=device, find_unused_parameters=False)
    total_num_batches = len(train_dataloader)
    for step, (batch_id, batch) in enumerate(train_dataloader):
        accumulate_scorer_training_data(step, batch_id[0], batch[0], model, 
                                        score_model, tokenizer, args, device, queue)

        if device == 0 and step % args.print_every == 0:
            scorer_acc_timer = timer_context.get_timer('scorer_data_acc_time')
            print()
            print(f"Rank: {device}:: Aggregated Batches:  {step}/{total_num_batches}. " + \
                        f"Avg step time: {scorer_acc_timer.avg_time():.3f}")

    if device == 0:
        print()
        scorer_acc_timer = timer_context.get_timer('scorer_data_acc_time')
        print(f"Rank: {device}:: Aggregated: {total_num_batches} items in " + \
                f"{scorer_acc_timer.cuml_time():.3f} seconds.")
    queue.close()

    cleanup()

def accumulate_scorer_training_data(step, batch_id, batch, model, 
                                      score_model, tokenizer, args, device, queue):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The perturbations are sampled from $\Deta \sim Q_{MGS}$.

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """
    data = {}
    with timer_context('scorer_data_acc_time') as ctxt_timer:
      start_time = timer()
      model = model.to(device=device)
      score_model = score_model.to(device)
      batch = batch.to(device=device)

      if batch.size(1) < args.context_length + 1:
          logging.error(
              f"Batch at step: {step} has sequences ({batch.size(1)})" + \
                f"shorter than the context length ({args.context_length})")
          return None

      inp, target = batch[:, :-1], batch[:, 1:]
      max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

      _, cur_decodings, cur_distances = ggs_utils.decode_and_distance(model, 
                                            tokenizer, batch, score_model, max_length, 
                                            device, args, average_distance=False)

      idx = f'accum_{batch_id}'
      data['idx'] = idx
      data['non_pert'] = (idx, InstanceType.NON_PERTURBED, batch_id, batch.clone(),
                            deepcopy(model), cur_decodings.clone(), cur_distances.clone())
      queue.put(data['non_pert'])

      # Get the current MLE gradients
      model_ = deepcopy(model)
      model_ = model_.to(device)
      inp, target = batch[:, :-1], batch[:, 1:]
      model_.eval()
      model_with_grad, _ = ggs_utils.mle_grad(model_, 
                            inp, target, args.pad_token_id, 
                            args.max_grad_norm)
      model_with_grad = model_with_grad.to(device)

      perturbed_models, log_rhos, noise_magnitudes, rng_states, perturb_types = \
            ggs_utils.perturb(model, model_with_grad, args.ggs_num_samples, 
                        args.ggs_noise, noise_scale=args.noise_scale,
                        zero_dist_only=args.zero_dist_only,
                        mle_dist_only=args.mle_dist_only,
                        include_mle_gradient=args.include_mle_gradient)

      data['pert'] = []
      for i, (p_model, log_rho, noise_mag, rng_state, perturb_type) in \
                enumerate(zip(perturbed_models, log_rhos,                                      
                        noise_magnitudes, rng_states, perturb_types)):
        p_model = p_model.to(device)
        _, per_decodings, per_distances = ggs_utils.decode_and_distance(p_model,
                                            tokenizer, batch, score_model, max_length,
                                            device, args, average_distance=False)
        data['pert'].append((idx, InstanceType.PERTURBED, batch_id, batch.clone(), deepcopy(model), 
            per_decodings.clone(), per_distances.clone(), log_rho, noise_mag, rng_state, perturb_type))
        queue.put(data['pert'][-1])

      end_time = timer()
      ctxt_timer.timeit(start_time, end_time)
    return data


def get_train_score_network_loss(data,
        score_network, tokenizer, device, args):

    dists_and_preds = []
    batch = data['batch']
    model_np = data['model']

    sequences_np, distances_np = data['non_pert']
    model_np = model_np.to(device=device)
    sequences_np = sequences_np.to(device)
    distances_np = distances_np.to(device)
    batch = batch.to(device=device)

    model_np.eval()
    pred_dist_np = score_network(model_np, batch, 
                        predictions=sequences_np)
    model_np.train()

    dists_and_preds.append((distances_np, pred_dist_np, None, InstanceType.NON_PERTURBED))
    
    # args.only_train_on_non_pert_data = True
    for model_p, sequences_p, distances_p, log_rho, noise_mag in \
                [] if args.only_train_on_non_pert_data else data['pert']:

        model_p = model_p.to(device=device)
        sequences_p = sequences_p.to(device)
        distances_p = distances_p.to(device)
        
        model_p.eval()
        pred_dist_p = score_network(model_p, batch, 
                            predictions=sequences_p)
        model_p.train()
        dists_and_preds.append((distances_p, pred_dist_p, log_rho, InstanceType.PERTURBED))
    losses = calculate_scorer_loss(dists_and_preds, args.scorer_loss_func, args.ggs_beta)
    return losses, dists_and_preds

def validate_score_network(valid_iter, score_network, tokenizer, device, args):
    cuml_valid_loss = 0.
    num_docs = 1
    if device == 0:
        valid_scorer_analysis = ScorerAnalysis('valid')
    score_network.eval()
    for step, data in enumerate(valid_iter):
        losses, distances_and_pred_distances = \
            get_train_score_network_loss(data,
                score_network, tokenizer, device, args)

        batch_loss = torch.sqrt(losses).sum().item()
        batch_size = losses.size(0)
        if batch_loss < 0:
            continue

        cuml_valid_loss += batch_loss
        num_docs += batch_size
        if device == 0:
            for distances, pred_distances, _, type in distances_and_pred_distances:
                pred_distances = pred_distances.squeeze(1).detach()
                valid_scorer_analysis(data['idx'], type, distances, pred_distances)

        if device == 0 and step % 5 == 0 and step > 0:
            print('Validation:: Step: %5d, Loss: %6.3f'
                  % (step, cuml_valid_loss / num_docs), end='\r')
    score_network.train()

    valid_info_dict = None 
    if device == 0:
        valid_info_dict = valid_scorer_analysis.get_metrics()

    return cuml_valid_loss/num_docs, valid_info_dict


class ScoreNetworkTrainerModule(LightningModule):
    def __init__(self, buffer, score_network, tokenizer, args, epochs):
        super().__init__()

        self.score_network = score_network
        self.buffer = buffer
        self.tokenizer = tokenizer
        self.args = args
        self.epochs = epochs
        self.valid_scorer_analysis = ScorerAnalysis('valid')
        self.train_scorer_analysis = ScorerAnalysis('train')

    def training_step(self, batch, batch_idx):
        losses, distances_and_pred_distances = \
            get_train_score_network_loss(batch_idx, batch, 
                self.score_network, self.tokenizer, self.device, self.args)

        # for distances, pred_distances, _, type in distances_and_pred_distances:
        #     pred_distances = pred_distances.squeeze(1).detach()
        #     self.train_scorer_analysis(batch_idx, type, distances, pred_distances)
        # train_info_dict = self.train_scorer_analysis.get_metrics()
        # train_info_dict['train/step_loss'] = losses.mean()
        # self.log_dict(train_info_dict, batch_size=1)
        return losses.mean() #{'loss': losses.mean()}

    def validation_step(self, batch, batch_idx):
        losses, distances_and_pred_distances = \
            get_train_score_network_loss(batch_idx, batch, 
                self.score_network, self.tokenizer, self.device, self.args)

        # for distances, pred_distances, _, type in distances_and_pred_distances:
        #     pred_distances = pred_distances.squeeze(1).detach()
        #     self.valid_scorer_analysis(batch_idx, type, distances, pred_distances)
        # valid_info_dict = self.valid_scorer_analysis.get_metrics()
        # valid_info_dict['valid/step_loss'] = losses.mean()
        # self.log_dict(valid_info_dict, batch_size=1)
        return losses.mean() #{'val_loss': losses.mean()}

    def training_epoch_end(self, outputs):
        info_dict = self.train_scorer_analysis.get_metrics()
        if self.trainer.is_global_zero:
            print(pformat(info_dict))
        # utils.log_tensorboard(info_dict, self.trainer.current_epoch)
        self.train_scorer_analysis.reset()

    def validation_epoch_end(self, outputs):
        info_dict = self.valid_scorer_analysis.get_metrics()
        if self.trainer.is_global_zero:
            print(pformat(info_dict))
        # utils.log_tensorboard(info_dict, self.trainer.current_epoch)
        self.valid_scorer_analysis.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.score_network.parameters(), lr=self.args.scorer_lr)
        return [optimizer], [StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)]

    def train_dataloader(self):
        train_dataset = ScoringNetworkTrainingDataset(self.buffer, 'train', self.args, shuffle=True, device=self.device)
        return DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=4)

    def val_dataloader(self):
        valid_dataset = ScoringNetworkTrainingDataset(self.buffer, 'valid', self.args, device=self.device)
        return DataLoader(valid_dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=4)


def train_score_network_v2(buffers, score_network, tokenizer, device, 
                  args, train_score_network_iteration=0, epochs=100):
    logging.info(f"Starting Score Network Training # {train_score_network_iteration}")
    logger = TensorBoardLogger("./debug/tb")

    score_network_module = ScoreNetworkTrainerModule(buffers, score_network, tokenizer, args, epochs)
    trainer = Trainer(gpus=-1, 
                    strategy=DDPPlugin(find_unused_parameters=False),
                    max_epochs=epochs, 
                    num_sanity_val_steps=0,
                    logger=logger)
    trainer.fit(score_network_module)


def train_score_network_v3(buffers, score_network, tokenizer, 
            args, train_score_network_iteration=0, epochs=100, world_size=1):
    world_size = torch.cuda.device_count()
    
    args.multigpu = True        
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_score_network_helper_v3, 
                args=(rank, buffers, score_network, tokenizer, 
                        args, train_score_network_iteration, epochs, world_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def train_score_network_helper_v3(device, buffers, score_network, tokenizer, 
            args, train_score_network_iteration=0, epochs=100, world_size=1):
    if args.multigpu:
        multigpu_setup(device, world_size)
    train_dataloader, valid_dataloader = buffers.get_dataloaders(args, device, world_size)
    score_network = score_network.to(device)
    score_network = DDP(score_network, device_ids=[device], 
                        output_device=device, find_unused_parameters=False)
            
    train_score_network(device, score_network, tokenizer, 
                        train_dataloader, valid_dataloader, args, 
                        train_score_network_iteration, epochs=epochs)
    cleanup()

def train_score_network(device, score_network, tokenizer, train_dataloader, valid_dataloader, 
                args, train_score_network_iteration=0, epochs=100):
    """ This method takes in the scoring data (B) and learns a parameterized scoring model (S) to 
        mimic the original cost function (C) by minimizing the L2 loss.
        $C$ is defined as
            C(\theta) = 1/|B| \sum_{i=1}^{B} c(y_i, F(x_i, \theta))

        The scoring function optimizes the following equation. 
            min_{W,b} \sum_{(x_i, y_i, yo_i, yp_i, \Delta_i) in B} || S(x_i, \theta, \Delta; W, b) - (C(\theta) - C(\theta + \Delta))||^2
            where S(x_i, \theta, \Delta; W, b) = W^T[R(x;\theta) - R(x;\theta+\Delta)] + b
    """
    min_valid_loss = math.inf
    best_score_network = None
    patience_counter = 0

    if device == 0:
        print('=' * 100)
        print('Start training the score network.\n')

    phi_optimizer = AdamW(score_network.parameters(), lr=args.scorer_lr)
    scheduler = ReduceLROnPlateau(phi_optimizer, 'min', patience=10, verbose=True)

    for epoch in range(epochs):
        score_network.train()
        if device == 0:
            train_scorer_analysis = ScorerAnalysis('train')
        cuml_train_loss = 0.
        num_docs = 1
        
        with timer_context('train_score_network_time') as ctxt_timer:
            start_time = timer()
            for step, data in enumerate(train_dataloader):
                phi_optimizer.zero_grad()
                
                losses, distances_and_pred_distances = \
                    get_train_score_network_loss(data,
                        score_network, tokenizer, device, args)

                batch_size = losses.size(0)
                loss = losses.sum()/batch_size

                if loss < 0:
                    continue
                
                if device == 0:
                    for distances, pred_distances, _, type in distances_and_pred_distances:
                        pred_distances = pred_distances.squeeze(1).detach()
                        train_scorer_analysis(data['idx'], type, distances, pred_distances)

                cuml_train_loss += torch.sqrt(losses).sum().item()
                num_docs += batch_size

                loss.backward()
                phi_optimizer.step()

                if device == 0 and step % 5 == 0 and step > 0:
                    print('Training:: Epoch: %3d :: Step: %5d, Loss: %6.3f' % (epoch, step, cuml_train_loss / num_docs),
                        end='\r')

            print()
            end_time = timer()
            if device == 0:
                train_info_dict = train_scorer_analysis.get_metrics()
                print(pformat(train_info_dict))

            ctxt_timer.timeit(start_time, end_time)

        with timer_context('validate_score_network_time') as ctxt_timer:
            start_time = timer()
            train_loss = cuml_train_loss / num_docs
            valid_loss, valid_info_dict = validate_score_network(
                valid_dataloader, score_network, tokenizer, device, args)

            if min_valid_loss < valid_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                min_valid_loss = valid_loss
                best_score_network = deepcopy(score_network)

            scheduler.step(valid_loss)
            # scheduler.step()

            if device == 0:
                print(pformat(valid_info_dict))

            if patience_counter > args.train_score_patience:
                print(f"Stopping Early at epoch: {epoch} with best validation loss: {min_valid_loss}")
                break

            end_time = timer()
            ctxt_timer.timeit(start_time, end_time)

        if device == 0:
            print('Epoch: %d :: Train Loss: %.3f, ' % (epoch, train_loss) +
                        'Best Valid Loss: %.3f, Valid Loss: %.3f, Epochs Since Last Best: %d '
                        % (min_valid_loss, valid_loss, patience_counter))
            print(f"Train score network epoch {epoch} done!")
            print(f"Avg Epoch Time: " +
                        f"Train: {timer_context('train_score_network_time').avg_time():.3f} " +
                        f"Valid: {timer_context('validate_score_network_time').avg_time():.3f}.")

        # prefix = f"tsn-{train_score_network_iteration}/"
        # valid_metrics = {
        #     prefix + "train_loss": train_loss,
        #     prefix + "valid_loss": valid_loss,
        #     prefix + "min_valid_loss": min_valid_loss,
        #     prefix + "epochs_since_best": patience_counter,
        #     prefix + "current_epoch": epoch,
        # }

        # for key, val in valid_info_dict.items():
        #     valid_metrics[prefix + key] = val

        # for key, val in train_info_dict.items():
        #     valid_metrics[prefix + key] = val

        # utils.log_tensorboard(valid_metrics, epoch)

    if device == 0:

        print('Done training the score network.\n')
        print('=' * 100)
        score_network = best_score_network

        if args.save_score_network:
            score_network_filepath = os.path.join(args.save_base_dir,
                                                'score_network.pkl')
            torch.save({
                'model_save_dict': score_network.state_dict(),
                'epochs': epoch,
                'dataset_size': len(train_dataloader),
            }, score_network_filepath)


def dagger_mgs_scoring_function(score_network, model, tokenizer, 
              batch, score_model, max_length, device, args, prefix):
    """ This methods takes in the batch and the model and
         returns the estimated scores for batch input according to the model.
    """
    with timer_context('dagger_mgs_scoring_func_time') as ctxt_timer:
      start_time = timer()
      
      outputs = torch.tensor([]).to(device)
      decoded = defaultdict(list)
      model.eval()
      pad = args.pad_token_id
      
      if not args.on_device:
          batch = batch.to(device=model.device)

      batched_distances = score_network(model, batch) \
                              .detach().cpu()

      # average across batch to compute c(\theta).
      distances = batched_distances.mean(dim=0).item()
      
      end_time = timer()
      ctxt_timer.timeit(start_time, end_time)
      model.train()
    return distances, outputs, decoded


def shall_accumulate_scorer_training_data(step, total_num_batches, args):
    # For first 25% of batches, aggregate data every batch.
    if step < total_num_batches // 4:
        return True

    # For best 25% of the batches, aggregate data every alternate batch.
    if step < total_num_batches // 2 and step % 2 == 0:
        return True

    # For last 50% of the batches, sample every fourth batch.
    if step % 4 == 0:
        return True

    return False

def add_args(parser):
    parser.add_argument(
        "--aggregated-data-size", type=int, default=2000,
    )
    parser.add_argument(
        "--aggregated-data-path", type=str,
    )
    parser.add_argument(
        "--save-aggregated-data", action='store_true',
    )
    parser.add_argument(
        "--use-saved-aggregated-data", action='store_true',
    )

    parser.add_argument('--include-mle-gradient', action='store_true')

    parser.add_argument(
        "--max-buffer-size", type=int, default=20000,
    )

    parser.add_argument(
        "--score-network-epochs", type=int, default=100,
    )
    parser.add_argument(
        "--retrain-score-network-epochs", type=int, default=30,
    )

    parser.add_argument(
        "--retrain-score-network-every", type=int, default=500
    )
    parser.add_argument(
        "--use-saved-score-network", action='store_true',
    )
    parser.add_argument(
        "--score-network-file", type=str,
    )
    parser.add_argument(
        "--save-score-network", action='store_true',
    )

    parser.add_argument('--efficient', action='store_true')

    parser.add_argument('--plot-times', action='store_true')

    parser.add_argument('--log-scoring-function', action='store_true')

    parser.add_argument('--on-device', action='store_true')

    parser.add_argument('--use-learned-scoring-function', 
                        action='store_true')

    parser.add_argument('--only-train-score-network', 
                        action='store_true')

    parser.add_argument('--only-train-on-non_pert_data', 
                        action='store_true')

    parser.add_argument(
        "--train-score-patience", type=int, default=30,
    )
    parser.add_argument(
        "--print-decodings", type=str, default=True,
    )
    parser.add_argument(
        "--heuristic", action='store_true',
    )

    parser.add_argument(
        "--use-sigmoid-for-scores", action='store_true',
    )

    parser.add_argument(
        "--scorer-lr", type=float, default=1e-5,
    )

    parser.add_argument(
        "--scorer-loss-func", type=str, default="mse",
    )

    parser.add_argument(
        "--initialize-score-network", action='store_true',
    )
    score_network_utils.add_args(parser)
    return parser