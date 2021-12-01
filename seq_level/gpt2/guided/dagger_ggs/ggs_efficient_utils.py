from collections import defaultdict, OrderedDict
from copy import deepcopy
from timeit import default_timer as timer
from enum import Enum

import hashlib
import logging
import math
import os
import random
import shelve
import torch
import torch.distributed as dist


from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import torch


import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils

timer_context = ggs_utils.TimerContext()

MODEL_ID = None

class InstanceType(Enum):
    PERTURBED=1
    NON_PERTURBED=2

def get_dataloader(args, dataset, rank=0, world_size=1, batch_size=1, collate_fn=None, shuffle=False):
    if args.multigpu:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    else:
        train_sampler = RandomSampler(dataset)

    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader


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
              f" DB size: {len(self.db):>6}",  end='\r')
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
                    'idx': idx,
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

    def _get_model_at_idx(self, idx2model_info, args, idx, device=None):
        data = idx2model_info[idx]

        idx2model = {}
        batch = self.db[data['batch_key']]
        model = self.db[data['model_key']]

        idx2model['idx'] = idx
        idx2model['batch'] = batch
        idx2model['model'] = model

        if device is not None:
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


def dagger_mgs_scoring_function(score_network, model, tokenizer, 
              batch, score_model, max_length, device, args, prefix):
    """ This methods takes in the batch and the model and
         returns the estimated scores for batch input according to the model.
    """
    with timer_context('dagger_mgs_scoring_func_time') as ctxt_timer:
      start_time = timer()
      is_original = False
      bpes = outputs = torch.tensor([]).to(device)
      decoded = defaultdict(list)
      model.eval()
      pad = args.pad_token_id
      
      if not args.on_device:
          batch = batch.to(device=model.device)

      batched_distances = score_network(model, batch) \
                              .detach().cpu()

      # average across batch to compute c(\theta).
      distances = batched_distances 
      
      end_time = timer()
      ctxt_timer.timeit(start_time, end_time)
      model.train()
    return is_original, distances, bpes, outputs, decoded


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
        "--max-buffer-size", type=int, default=20000,
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
        "--print-decodings", type=str, default=True,
    )
    parser.add_argument(
        "--heuristic", action='store_true',
    )

    parser.add_argument(
        "--use-sigmoid-for-scores", action='store_true',
    )


    return parser