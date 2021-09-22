import torch
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Config
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler
import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils
import seq_level.gpt2.train as train_utils
import os
from functools import partial
from seq_level.gpt2.guided.metrics import GuidedMetrics
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

from timeit import default_timer as timer
import pandas as pd
import collections
import random
import shelve
import hashlib
import pickle
import logging
import math
import scipy.stats as stats


def _hash_tensor(obj):
    return hashlib.sha1(bytes(obj.cpu().numpy())).hexdigest()


def _hash_model(model):
    return hashlib.sha1(next(model.parameters()).detach().cpu().numpy()).hexdigest()


MODEL_ID = None


class RingBuffer:
    def __init__(self, max_size=1000, persistence='none', persistent_file_path=None, shuffle=True, iter_device=None,
                 on_device=False):
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
        # self.executor = ThreadPoolExecutor(max_workers=6)

    def __len__(self):
        return len(self.train_queue)

    def append(self, idx, type, batch_id, batch, model, sequences, 
                distances, rng_state=None, apply_to_mle_grad=None):

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

        print(f"Id: {idx}::" +
              f" Queue Sizes: {len(self.train_queue)}/{len(self.valid_queue)}," +
              f" DB size: {len(self.db)}", end='\r')

        if len(queue) >= queue_max_size:
            (_, _, old_batch_key, old_model_key,
             _, _, _, _) = queue.pop(0)
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

        # batch_key = f"batch_{_hash_tensor(batch)}"
        batch_key = f"batch_{batch_id}"
        if batch_key not in self.db:
            if not self.on_device:
                batch = deepcopy(batch).to(device=torch.device("cpu"))
            self.db[batch_key] = batch
            self.db_counter[batch_key] = 0
        self.db_counter[batch_key] += 1

        global MODEL_ID
        model_key = f"model_{MODEL_ID}"
        if model_key not in self.db:
            if not self.on_device:
                model = deepcopy(model).to(device=torch.device("cpu"))
            self.db[model_key] = model
            self.db_counter[model_key] = 0
        self.db_counter[model_key] += 1

        sequences_key = None

        if not self.on_device:
            distances = distances.cpu()

        queue.append((idx, type, batch_key, model_key, 
                        sequences_key, distances, rng_state, 
                        apply_to_mle_grad))

    def get_iterators(self, shuffle=True):
        def _batch_generator(iterable, shuffle=True):
            if shuffle:
                iterable = random.sample(iterable, len(iterable))

            for (idx, type, batch_key, model_key, sequences_key, distances, rng_state, apply_to_mle_grad) in iterable:
                batch = self.db[batch_key].type(torch.long)
                sequences = None
                model = self.db[model_key]

                if distances.size(0) != batch.size(0):
                    logging.error(
                        f"Distance: {distances.size(0)}, Batch: ({batch.size()}), {batch_key} Sequence: {sequences_key}" + \
                        f"Model: {model_key}.")
                    continue
                yield (idx, type, batch, model, sequences, distances, rng_state, apply_to_mle_grad)

        return _batch_generator(self.train_queue, shuffle), _batch_generator(self.valid_queue, False)


total_scoring_time = {
    "cuml": 0,
    "tick": 0,
}

curr_scoring_time = {
    "cuml": 0,
    "tick": 0,
}

mle_grad_computation_time = {
    "cuml": 0,
    "tick": 0,
}

perturb_computation_time = {
    "cuml": 0,
    "tick": 0,
    'num_perturb': 0,
}

perturb_scoring_time = {
    "cuml": 0,
    "tick": 0,
}

weight_computation_time = {
    "cuml": 0,
    "tick": 0,
}

ggs_update_time = {
    "cuml": 0,
    "tick": 0,
}

metrics_update_time = {
    "cuml": 0,
    "tick": 0,
}

total_mgs_time = {
    "cuml": 0,
    "tick": 0,
}

total_train_step_time = {
    "cuml": 0,
    "tick": 0,
}

train_score_network_time = {
    'cuml': 0,
    'tick': 0,
}

aggregation_step_time = {
    'cuml': 0,
    'tick': 0,
}


def accumulate_score_function_training_data(step, batch_id, batch, buffer, model, score_model, tokenizer, args, device):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The perturbations are sampled from $\Deta \sim Q_{MGS}$.

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """
    batch.squeeze_(0)
    batch = batch.to(device=device)
    if batch.size(1) < args.context_length + 1:
        logging.error(
            f"Batch at step: {step} has sequences: {batch.size(1)} shorter than the context length: {args.context_length}")
        return buffer

    inp, target = batch[:, :-1], batch[:, 1:]
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)
    model = model.to(device=device)
    model.eval()
    _, cur_decodings, cur_distances = ggs_utils.decode_and_distance(
        model, tokenizer, batch, score_model, max_length, device, args, average_distance=False
    )

    idx = f'accum_{step}'
    buffer.append(idx, 'current', batch_id, batch, model, cur_decodings, cur_distances)

    # Get the current MLE gradients
    model.train()
    per_model, rng_state, apply_to_mle_grad = perturb(model, batch, step, tokenizer, args, device=device)

    _, per_decodings, per_distances = ggs_utils.decode_and_distance(
        per_model, tokenizer, batch, score_model, max_length, device, args, average_distance=False
    )
    buffer.append(idx, 'pertubed', batch_id, batch, model, 
                    per_decodings, per_distances, 
                    rng_state=rng_state, apply_to_mle_grad=apply_to_mle_grad)
    return buffer


def perturb(model, batch, step, tokenizer, args, device=None,
                rng_state=None,  apply_to_mle_grad=None):
    per_model = deepcopy(model)
    inp, target = batch[:, :-1], batch[:, 1:]

    apply_to_mle_grad = apply_to_mle_grad or (random.random() < 0.5)

    model_with_grad, _ = ggs_utils.mle_grad(
        per_model, inp, target, tokenizer.pad_token_id, args.max_grad_norm
    )
    model_with_grad_param_dict = dict(model_with_grad.named_parameters())

    with ggs_utils.RNG(rng_state, device) as (rng, rng_state):
        for name, param in per_model.named_parameters():
            perturbation = torch.randn(param.size(), generator=rng, device=param.device)

            param_with_grad = model_with_grad_param_dict[name]
            gradient = -param_with_grad.grad.data

            if args.noise_scale == 'uniform':
                noise_ = args.ggs_noise * perturbation * (
                            gradient.data.abs().sum() / gradient.data.numel())
            else:
                noise_ = args.ggs_noise * perturbation

            # TODO: Should we consider random noise addition for data aggregation and
            # score training. 
            if apply_to_mle_grad:
                epsilon = noise_ + gradient
            else:
                epsilon = noise_

            param.data = param.data + epsilon
    return per_model, rng_state, apply_to_mle_grad


def get_train_score_network_loss(idx, type, model, batch, 
                distances, score_network, tokenizer, device):
    model = model.to(device=device)
    model.eval()

    batch = batch.to(device=device)
    pad = tokenizer.pad_token_id

    outputs = score_network(model, batch, pad)

    distances = distances.to(device=device)
    if distances.size(0) != outputs.size(0):
        logging.error(f"Batch: ({idx} {type} {batch.size()}) != Distance {distances.size()}")
        return -1.0

    loss = F.mse_loss(
        outputs,
        distances.view(-1, 1),
        reduction='sum',
    )
    return loss, outputs


def validate_score_network(valid_iter, score_network, tokenizer, device, args):
    cuml_valid_loss = 0.
    num_docs = 0
    cuml_perturbed_loss = 0.
    cuml_non_perturbed_loss = 0.

    cuml_score_func_fit_all = 0.
    cuml_score_func_fit_perturbed = 0.
    cuml_score_func_fit_original = 0.

    num_docs_perturbed = 0
    num_docs_non_perturbed = 0

    # idx => {'original': (true_dist, pred_dist), 'perturbed': [(true_dist, pred_dist)]}
    score_func_diff_fit_dict = {}

    true_distances = []
    predicted_distances = []

    true_distances_perturbed = []
    true_distances_non_perturbed = []
    predicted_distances_perturbed = []
    predicted_distances_non_perturbed = []

    for step, (idx, type, batch, model, _, distances, rng_state,
                        apply_to_mle_grad) in enumerate(valid_iter):

        if type == "pertubed":
            model, _, _ = perturb(model, batch, idx, tokenizer, 
                           args, device=device, rng_state=rng_state,
                           apply_to_mle_grad=apply_to_mle_grad)

        loss, pred_distances = get_train_score_network_loss(idx, 
                                    type, model, batch, distances, 
                                    score_network, tokenizer, device)

        if loss < 0:
            continue

        true_distances += distances.tolist()
        pred_distances = pred_distances.squeeze(1).detach()
        predicted_distances += pred_distances.tolist()
        score_func_fit = torch.sum(torch.abs(distances - pred_distances)/pred_distances).item()
        
        cuml_valid_loss += loss.item()
        cuml_score_func_fit_all += score_func_fit

        num_docs += batch.size(0)

        if idx not in score_func_diff_fit_dict:
            score_func_diff_fit_dict[idx] = {'original': None, 
                                             'perturbed': [],}
        
        if type == "pertubed":
            num_docs_perturbed += batch.size(0)
            cuml_perturbed_loss += loss.item()
            true_distances_perturbed += distances.tolist()
            predicted_distances_perturbed += pred_distances.tolist()
            cuml_score_func_fit_perturbed = score_func_fit
            score_func_diff_fit_dict[idx]['perturbed'].append((distances.mean(), pred_distances.mean()))
        else:
            num_docs_non_perturbed += batch.size(0)
            cuml_non_perturbed_loss += loss.item()
            true_distances_non_perturbed += distances.tolist()
            predicted_distances_non_perturbed += pred_distances.tolist()
            cuml_score_func_fit_original += score_func_fit
            score_func_diff_fit_dict[idx]['original'] = (distances.mean(), pred_distances.mean())

        if step % 5 == 0 and step > 0:
            print('Validation:: Step: %d, Loss: %.2f'
                  % (step, cuml_valid_loss / num_docs), end='\r')
    print()

    cuml_score_func_diff_fit = 0.
    func_diff_fit_count = 0
    for idx, diff_dict in score_func_diff_fit_dict.items():
        original_true_score, original_pred_score = diff_dict['original']
        pertured_scores = diff_dict['perturbed']

        for perturb_true_score, perturb_pred_score in pertured_scores:
            if (perturb_true_score - original_true_score) == 0:
                logging.debug(f"For idx: {idx}," + 
                                " perturbed score: {perturb_true_score}" + 
                                " original score: {original_true_score}" + 
                                " are equal.")
                continue

            cuml_score_func_diff_fit += torch.abs((perturb_pred_score - original_pred_score)/(perturb_true_score - original_true_score)).item()
            func_diff_fit_count += 1

    return cuml_valid_loss / num_docs, {"all_corr": stats.kendalltau(true_distances, predicted_distances)[0],
                                        "perturbed_corr":
                                            stats.kendalltau(true_distances_perturbed, predicted_distances_perturbed)[
                                                0],
                                        "original_corr": stats.kendalltau(true_distances_non_perturbed,
                                                                          predicted_distances_non_perturbed)[0],
                                        "perturbed_loss": cuml_perturbed_loss / num_docs_perturbed,
                                        "original_loss": cuml_non_perturbed_loss / num_docs_non_perturbed, 
                                        "score_func_fit_all": cuml_score_func_fit_all / num_docs,
                                        "score_func_fit_perturbed": cuml_score_func_fit_perturbed / num_docs_perturbed,
                                        "score_func_fit_original": cuml_score_func_fit_original / num_docs_non_perturbed,
                                        "score_func_diff_fit": cuml_score_func_diff_fit / func_diff_fit_count,}


def train_score_network(buffers, score_network, tokenizer, device, 
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
    print('=' * 100)
    print('Start training the score network.\n')
    score_network.train()
    score_network = score_network.to(device=device)

    phi_optimizer = optim.Adam(score_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(phi_optimizer, step_size=20, gamma=0.5, verbose=True)

    for epoch in range(epochs):
        train_iterator, valid_iterator = buffers.get_iterators()
        cuml_train_loss = 0.
        num_docs = 0
        train_score_network_start = timer()
        for step, (idx, type, batch, model, _, distances, rng_state,
             apply_to_mle_grad) in enumerate(train_iterator):

            phi_optimizer.zero_grad()
            if type == "pertubed":
                model, _, _ = perturb(model, batch, idx, tokenizer, args, 
                                device=device, rng_state=rng_state, apply_to_mle_grad=apply_to_mle_grad)

            loss, _ = get_train_score_network_loss(idx, 
                        type, model, batch, distances, score_network, tokenizer, device)

            if loss < 0:
                continue

            cuml_train_loss += loss.item()
            num_docs += batch.size(0)

            loss.backward()
            phi_optimizer.step()

            if step % 5 == 0 and step > 0:
                print('Training:: Epoch: %d :: Step: %d, Loss: %.2f' % (epoch, step, cuml_train_loss / num_docs),
                      end='\r')

            if not args.on_device:
                # Move model back to CPU so that it doesn't hog GPU
                # memory as it will not be removed from the context.
                model.to(device=torch.device("cpu"))
                distances.to(device=torch.device("cpu"))

        print()
        train_loss = cuml_train_loss / num_docs
        valid_loss, valid_info_dict = validate_score_network(
                                        valid_iterator, score_network,
                                        tokenizer, device, args)

        if min_valid_loss < valid_loss:
            patience_counter += 1
        else:
            patience_counter = 0
            min_valid_loss = valid_loss
            best_score_network = deepcopy(score_network)
            logging.info(pformat(valid_info_dict))

        if patience_counter > args.train_score_patience:
            logging.info(f"Stopping Early at epoch: {epoch} with best validation loss: {min_valid_loss}")
            break

        scheduler.step()
        train_score_network_end = timer()
        train_score_network_time['cuml'] += train_score_network_end - train_score_network_start
        train_score_network_time['tick'] += 1
        logging.info('Epoch: %d :: Train Loss: %.2f, ' % (epoch, train_loss) +
                     'Best Valid Loss: %.2f, Valid Loss: %.2f, Epochs Since Last Best: %d '
                     % (min_valid_loss, valid_loss, patience_counter))
        logging.info(f"Train score network epoch {epoch} done!")
        logging.info(f"Avg Epoch Time: {train_score_network_time['cuml'] / train_score_network_time['tick']}")

        prefix = f"train_score_network_{train_score_network_iteration}/"
        valid_metrics = {
            prefix + "train_loss": train_loss,
            prefix + "valid_loss": valid_loss,
            prefix + "min_valid_loss": min_valid_loss,
        }

        for key, val in valid_info_dict.items():
            valid_metrics[prefix + key] = val
        utils.log_tensorboard(valid_metrics, epoch)

    print('Done training the score network.\n')
    print('=' * 100)
    score_network = best_score_network

    if args.save_score_network:
        score_network_filepath = os.path.join(args.save_base_dir,
                                              'score_network.pkl')
        torch.save({
            'model_save_dict': score_network.state_dict(),
            'epochs': epoch,
            'dataset_size': len(buffers),
        }, score_network_filepath)


def original_mgs_scoring_function(buffer, is_target_function, model, tokenizer, batch, score_model, max_length, device,
                                  args, prefix):
    decoded = defaultdict(list)
    bpes_curr, outputs, distance_curr = ggs_utils.decode_and_distance(
        model, tokenizer, batch, score_model,
        max_length, device, args, average_distance=False)

    # Keeping this commented for the time being as need to figure out
    # how to integrate rng_state caching.
    if False and args.efficient and is_target:
        idx = f'mgs_{args.log_step}'
        buffer.append(idx, prefix, batch_id, batch, model,
                      outputs, distance_curr)

    if not isinstance(batch, list):
        outputs = outputs.tolist()

    for i, decoding in enumerate(outputs):
        if tokenizer.eos_token_id in decoding:
            decoding = decoding[:decoding.index(tokenizer.eos_token_id) + 1]
        decoded[f'{prefix}_{i}'].append(tokenizer.decode(decoding))
    return distance_curr.mean().item(), bpes_curr, decoded


def dagger_mgs_scoring_function(score_network, model, tokenizer, 
            batch, score_model, max_length, device, args, prefix):
    """ This methods takes in the batch and the model and
         returns the estimated scores for batch input according to the model.
    """
    outputs = torch.tensor([])
    decoded = defaultdict(list)
    model.eval()
    pad = tokenizer.pad_token_id
    batch = batch.to(device=model.device)

    batched_distances = score_network(model, batch, pad) \
                            .detach().cpu()

    # average across batch to compute c(\theta).
    distances = batched_distances.mean(dim=0).item()

    return distances, outputs, decoded


def MGS(batch, model, score_model, tokenizer, args, device, metrics, optimizer,
        scoring_function=None,
        target_scoring_func=None):
    """
    MGS algorithm parameterized to work in original as well as efficient mode.
    """
    distance_comp = []
    mgs_time_start = timer()

    inp, target = batch[:, :-1].to(device=device), batch[:, 1:].to(device=device)

    # -- Decode with current model (required for computing the 'weights' later).
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

    decoded = defaultdict(list)

    curr_scoring_start = timer()
    distance_curr, bpes_curr, decoded_samples = scoring_function(
        model, tokenizer, batch, score_model, max_length, device, args, prefix='original'
    )

    if args.efficient and args.log_scoring_function and args.log_step % args.print_every == 1:
        distance_curr_score, _, _ = target_scoring_func(
            model, tokenizer, batch, score_model, max_length, device, args, prefix='original'
        )
        distance_comp.append(('original', distance_curr, distance_curr_score))
        logging.info(f"Distances: original: C => {distance_curr} C_t => {distance_curr_score}")

    decoded.update(decoded_samples)
    curr_scoring_end = timer()
    curr_scoring_time['cuml'] += curr_scoring_end - curr_scoring_start
    curr_scoring_time['tick'] += 1

    total_scoring_time['cuml'] += curr_scoring_end - curr_scoring_start
    total_scoring_time['tick'] += 1

    # -- Obtain MLE gradients
    mle_grad_computation_start = timer()
    model_ = deepcopy(model)
    model_with_grad, mle_loss = ggs_utils.mle_grad(
        model_, inp, target, tokenizer.pad_token_id, args.max_grad_norm
    )
    mle_grad_computation_end = timer()
    mle_grad_computation_time['cuml'] += mle_grad_computation_end - mle_grad_computation_start
    mle_grad_computation_time['tick'] += 1

    perturb_computation_start = timer()
    # -- Perturb

    if args.heuristic:
        perturbed_models, log_rhos, noise_magnitudes = ggs_utils.heuristic_perturb(
            model, model_with_grad, args.ggs_num_samples, args.ggs_noise,
            tokenizer, batch, score_model, device, args,
            distance_curr_score,
            noise_scale=args.noise_scale,
            zero_dist_only=args.zero_dist_only,
            mle_dist_only=args.mle_dist_only,
            include_mle_gradient=args.include_mle_gradient
        )
    else:
        perturbed_models, log_rhos, noise_magnitudes = ggs_utils.perturb(
            model, model_with_grad, args.ggs_num_samples, args.ggs_noise,
            noise_scale=args.noise_scale,
            zero_dist_only=args.zero_dist_only,
            mle_dist_only=args.mle_dist_only,
            include_mle_gradient=args.include_mle_gradient
        )
    perturb_computation_end = timer()
    perturb_computation_time['cuml'] += perturb_computation_end - perturb_computation_start
    perturb_computation_time['tick'] += 1

    perturb_scoring_start = timer()
    # -- Decode with perturbed models and compute task metric
    distances = []
    for i, p_model in enumerate(perturbed_models):

        distance, _, decoded_samples = scoring_function(p_model, tokenizer, batch, score_model,
                                                        max_length, device, args, prefix=f'preturb_{i}'
                                                        )
        if args.efficient and args.log_scoring_function and args.log_step % args.print_every == 1:
            distance_score, _, _ = target_scoring_func(p_model, tokenizer, batch,
                                                       score_model, max_length, device, args, prefix='preturb_{i}')
            distance_comp.append(('preturb_{i}', distance, distance_score))
            logging.info(f"Distances: preturb_{i}: C => {distance} C_t => {distance_score}")
            logging.info(
                f"C'_{i} - C: => {distance - distance_curr} C'_t_{i} - C_t => {distance_score - distance_curr_score}")

        distances.append(distance)
        decoded.update(decoded_samples)
    perturb_scoring_end = timer()
    perturb_scoring_time['cuml'] += perturb_scoring_end - perturb_scoring_start
    total_scoring_time['cuml'] += perturb_scoring_end - perturb_scoring_start
    perturb_scoring_time['tick'] += 1

    # -- Compute weights
    # Kushal: please revise score_network(distance_curr - distances), where the score_function's output is embedding.

    weight_computation_start = timer()
    log_weights = ggs_utils.compute_weight(distance_curr, distances, log_rhos, args.ggs_beta)

    # -- Compute weighted average of the directions
    update_directions = ggs_utils.parameter_weighted_average(
        model, perturbed_models, log_weights
    )
    weight_computation_end = timer()
    weight_computation_time['cuml'] += weight_computation_end - weight_computation_start
    weight_computation_time['tick'] += 1

    ggs_update_start = timer()
    # -- Perform update
    global MODEL_ID
    MODEL_ID = ggs_utils.update(model, update_directions, optimizer, args.max_grad_norm)
    ggs_update_end = timer()
    ggs_update_time['cuml'] += ggs_update_end - ggs_update_start
    ggs_update_time['tick'] += 1

    metrics_update_start = timer()
    # -- Record statistics
    metrics.step(
        mle_loss.item(), distance_curr, bpes_curr, target, args.context_length,
        tokenizer.pad_token_id, tokenizer.eos_token_id,
        model_with_grad, update_directions, log_rhos, log_weights,
        noise_magnitudes, distances
    )
    metrics_update_end = timer()
    metrics_update_time['cuml'] += metrics_update_end - metrics_update_start
    metrics_update_time['tick'] += 1

    mgs_time_end = timer()
    total_mgs_time['cuml'] += mgs_time_end - mgs_time_start
    total_mgs_time['tick'] += 1
    return decoded


def shall_accumulate_score_function_training_data(step, total_num_batches, args):
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


class ScoreNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(ScoreNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, model, batch, pad):
        device = batch.device
        mask = batch.ne(pad).float().to(device=device)
        batch_ = deepcopy(batch).to(device=device)
        batch_[batch == pad] = 0

        model_output = model(batch_,
                            attention_mask=mask,
                            output_hidden_states=True)

        emb = model_output \
                .hidden_states[-1][:, -1, :] \
                .detach()
        output = self.fc(emb)
        return output


def train(model, tokenizer, dataset_tensor_dict, args, device):
    global MODEL_ID
    MODEL_ID = ggs_utils.get_model_id(model)

    score_network_training_iter = 0
    model.train()
    train_sampler = RandomSampler(dataset_tensor_dict['train'])
    train_dataloader = DataLoader(
        dataset_tensor_dict['train'],
        sampler=train_sampler,
        batch_size=1
    )

    total_num_batches = len(train_dataloader)
    optimizer, scheduler = utils.get_optimizer(model, total_num_batches, args)
    best_val_loss = 10000
    patience = args.patience
    stats_cache = defaultdict(list)
    average_times = {}
    score_model = deepcopy(model)

    scoring_function = partial(original_mgs_scoring_function, None, False)
    target_scoring_func = None

    config = GPT2Config()
    score_network = ScoreNetwork(input_size=config.hidden_size) \
                        .to(device=device)

    if args.efficient:
        # Initialize buffer and pretrain score network.
        print('=' * 100)

        # If using saved aggregated data, use it, else, initialize an empty buffer.
        if args.use_saved_aggregated_data:
            with open(args.aggregated_data_path, 'rb') as aggregated_datafile:
                buffer = pickle.load(aggregated_datafile)
            logging.info(f"Loading Aggregated data from {args.aggregated_data_path}. Size: {len(buffer)}")
        else:
            buffer = RingBuffer(max_size=args.max_buffer_size,
                                persistence='none',
                                persistent_file_path=os.path.join(
                                    args.save_base_dir,
                                    "persistence_datastore"),
                                on_device=args.on_device)


        initial_training_epochs = args.score_network_epochs
        # If using saved score network, use it, else accumulate training data, 
        # and train network on the accumulated data.
        if args.use_saved_score_network:
            checkpoint = torch.load(args.score_network_file)
            score_network.load_state_dict(
                checkpoint['model_save_dict'])
            epochs_trained = score_model_checkpoint['epochs']
            logging.info(f"Loading Phi Network trained for {epochs_trained} epochs from {args.score_network_file}.")
            initial_training_epochs = args.retrain_score_network_epochs
        else:
            if not args.use_saved_aggregated_data:
                logging.info("Started Initial Data Aggregation.")
                for step, (batch_id, batch) in enumerate(train_dataloader):
                    if step >= args.aggregated_data_size:
                        break

                    aggregate_step_start = timer()
                    accumulate_score_function_training_data(step, batch_id, batch,
                                buffer, model, score_model, tokenizer, args, device)

                    aggregate_step_end = timer()
                    aggregation_step_time['cuml'] += aggregate_step_end - aggregate_step_start
                    aggregation_step_time['tick'] += 1
                    if step % args.print_every == 0:
                        logging.info(f"Aggregated Batches:  {step}/{total_num_batches}." +
                                     f"Avg time: {aggregation_step_time['cuml'] / aggregation_step_time['tick']}")

                logging.info(f"Aggregated: {step * 2} items in {aggregation_step_time['cuml']} seconds.")

                if args.save_aggregated_data:
                    buffer_filepath = os.path.join(args.save_base_dir, 'buffer.pkl')
                    logging.info(f"Saving Aggregated Data at {buffer_filepath}")
                    with open(buffer_filepath, 'wb') as buffer_file:
                        pickle.dump(buffer, buffer_file)

        logging.info("Training Scoring Network on Aggregated Data.")
        train_score_network(buffer,score_network,tokenizer, device,
                            args, score_network_training_iter,
                            epochs=initial_training_epochs)

        scoring_function = partial(original_mgs_scoring_function, buffer, False)
        target_scoring_func = partial(dagger_mgs_scoring_function, score_network)

        if args.use_learned_scoring_function:
            scoring_function = partial(dagger_mgs_scoring_function, score_network)
            target_scoring_func = partial(original_mgs_scoring_function, buffer, False)
        print('=' * 100)

    for epoch_number in range(args.num_train_epochs):
        metrics = GuidedMetrics()

        for step, (batch_id, batch) in enumerate(train_dataloader):
            if args.efficient:
                if shall_accumulate_score_function_training_data(step, total_num_batches, args):
                    accumulate_score_function_training_data(step, batch_id, batch,
                                buffer, model, score_model, tokenizer, args, device) 

                if (step + 1) % args.retrain_score_network_every == 0:
                    score_network_training_iter += 1
                    train_score_network(buffer, score_network,
                        tokenizer, device, args, 
                        score_network_training_iter,
                        epochs=args.retrain_score_network_epochs,)

            train_step_time_start = timer()

            if len(batch.shape) < 2:
                logging.error(f"Batch has a single item and is of shape: {batch.shape}")
                continue

            if len(batch.shape) > 2:
                batch = batch.squeeze(0)

            if batch.size(-1) < args.context_length + 1:
                logging.error(
                    f"Batch at step: {step} has sequences: {batch.size(1)} shorter than the context length: {args.context_length}")
                continue

            batch = batch.to(device=device)

            decoded = MGS(batch=batch,
                          model=model,
                          score_model=score_model,
                          tokenizer=tokenizer,
                          args=args,
                          device=device,
                          metrics=metrics,
                          optimizer=optimizer,
                          scoring_function=scoring_function,
                          target_scoring_func=target_scoring_func
                          )
            train_step_time_end = timer()

            total_train_step_time['cuml'] += train_step_time_end - train_step_time_start
            total_train_step_time['tick'] += 1

            if step % args.print_every == 0:
                metrics_ = metrics.normalize('train')
                metrics.reset()
                logging.info("Epoch %d   \t Step %d   \tmle: %.3f\tdist: %.3f\tnon_term: %.3E\tmle_weight: %.3E" % (
                    epoch_number,
                    step,
                    metrics_['train/mle_loss'],
                    metrics_['train/distance'],
                    metrics_['train/non_term'],
                    metrics_['train/model/mle_weight']
                ))

                utils.log_tensorboard(metrics_, args.log_step)
                stats_cache['train/mle_loss'].append(metrics_['train/mle_loss'])
                stats_cache['train/distance'].append(metrics_['train/distance'])

                average_times['total_scoring_time'] = total_scoring_time['cuml'] / total_scoring_time['tick']
                average_times['curr_scoring_time'] = curr_scoring_time['cuml'] / curr_scoring_time['tick']
                average_times['mle_grad_computation_time'] = mle_grad_computation_time['cuml'] / \
                                                             mle_grad_computation_time['tick']
                average_times['perturb_computation_time'] = perturb_computation_time['cuml'] / perturb_computation_time[
                    'tick']
                average_times['perturb_scoring_time'] = perturb_scoring_time['cuml'] / perturb_scoring_time['tick']
                average_times['weight_computation_time'] = weight_computation_time['cuml'] / weight_computation_time[
                    'tick']
                average_times['ggs_update_time'] = ggs_update_time['cuml'] / ggs_update_time['tick']
                average_times['metrics_update_time'] = metrics_update_time['cuml'] / metrics_update_time['tick']
                average_times['total_mgs_time'] = total_mgs_time['cuml'] / total_mgs_time['tick']
                average_times['total_train_step_time'] = total_train_step_time['cuml'] / total_train_step_time['tick']

                if args.plot_times:
                    df = pd.DataFrame.from_dict(average_times,
                                                orient='index',
                                                columns=['avg. time'])
                    print(df)

                if not args.use_learned_scoring_function and args.print_decodings:
                    i = random.choice(range(batch.size(0)))
                    print(f"Step: {step}")
                    print(f"theta(x_{i}): {decoded['original_%d' % i][0]}")
                    for j in range(args.ggs_num_samples):
                        print(f"theta+Delta_{j}(x_{i}): {decoded['preturb_%d_%d' % (j, i)][0]}")
                    print('\n')

            if args.log_step % args.valid_every == 0:
                val_loss, val_metrics, decodings = train_utils.valid_iteration(
                    dataset_tensor_dict['valid'], model, score_model, train_utils.get_mle_loss, tokenizer, device,
                    context_length=args.eval_context_length,
                    num_decodings=250,
                    args=args
                )

                if args.print_decodings:
                    logging.info(f"Validation Decodings at Step: {step}")
                    prefixes = decodings['text_prefix']
                    sentences = decodings['text_decoding_including_prefix']

                    to_print_idxs = random.sample(range(len(sentences)), 10)
                    for i in to_print_idxs:
                        print(f"{'Prefix':<10}: {prefixes[i]}")
                        print(f"{'Sequence':<10}: {sentences[i]}")                    
                        print('\n')

                if args.save_all == 1:
                    save_dir = os.path.join(args.save_base_dir, str(args.log_step))
                    os.makedirs(save_dir)
                    utils.save(model, save_dir)
                if val_metrics['valid/distance-%s' % args.ggs_metric] < best_val_loss:
                    logging.info('Best distance achieved (%.5f)' % val_metrics['valid/distance-%s' % args.ggs_metric])
                    if not args.no_checkpoint:
                        utils.save(model, args.save_base_dir)
                    utils.save_metrics(val_metrics, args)
                    utils.save_decodings(decodings, args)
                    best_val_loss = val_metrics['valid/distance-%s' % args.ggs_metric]
                    patience = args.patience
                else:
                    patience = patience - 1

                model.train()
                utils.log_tensorboard(val_metrics, args.log_step)

                if patience == 0:
                    return

            if args.max_train_steps > 0 and args.log_step >= args.max_train_steps:
                return

            args.log_step += 1
            torch.cuda.empty_cache()


def add_args(parser):
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--ggs-noise", type=float, default=1.0)
    parser.add_argument("--ggs-beta", type=float, default=100.0)
    parser.add_argument("--ggs-num-samples", type=int, default=4)
    parser.add_argument("--decode-len-multiplier", type=float, default=1.3)
    parser.add_argument(
        "--ggs-metric",
        choices=['edit', 'lm'],
        default='lm'
    )
    parser.add_argument(
        "--bleu-smoothing",
        choices=['method%d' % i for i in range(1, 8)],
        default='method2'
    )
    parser.add_argument(
        "--noise-scale", choices=['uniform', 'constant'], default='uniform'
    )
    parser.add_argument(
        "--zero-dist-only", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--mle-dist-only", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--save-all", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=-1
    )

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
        "--max-buffer-size", type=int, default=4000,
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

    parser.add_argument('--use-learned-scoring-function', action='store_true')

    parser.add_argument(
        "--train-score-patience", type=int, default=20,
    )
    parser.add_argument(
        "--print-decodings", type=str, default=True,
    )
    parser.add_argument(
        "--heuristic", action='store_true',
    )
    return parser
