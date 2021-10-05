from collections import defaultdict
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
import torch.nn.functional as F
import torch.optim as optim

import seq_level.gpt2.guided.score_network as score_network_utils
import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils
timer_context = ggs_utils.TimerContext()
def _hash_tensor(obj):
    return hashlib.sha1(bytes(obj.cpu().numpy())).hexdigest()


def _hash_model(model):
    return hashlib.sha1(next(model.parameters()).detach().cpu().numpy()).hexdigest()

MODEL_ID = None

class InstanceType(Enum):
    PERTURBED=1
    NON_PERTURBED=2

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
                distances, rng_state=None, perturb_type=None):

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
             old_seq_key, _, _, _) = queue.pop(0)
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

        sequences_key =  f"seq_{batch_id}_{MODEL_ID}"
        if sequences_key not in self.db:
            if not self.on_device:
                batch = deepcopy(sequences).to(device=torch.device("cpu"))
            self.db[sequences_key] = sequences
            self.db_counter[sequences_key] = 0
        self.db_counter[sequences_key] += 1

        if not self.on_device:
            distances = distances.cpu()

        queue.append((idx, type, batch_key, model_key, 
                        sequences_key, distances, rng_state, 
                        perturb_type))

    def get_iterators(self, shuffle=True):
        def _batch_generator(iterable, shuffle=True):
            if shuffle:
                iterable = random.sample(iterable, len(iterable))

            for (idx, type, batch_key, model_key, sequences_key, 
                    distances, rng_state, perturb_type) in iterable:

                batch = self.db[batch_key]
                sequences = self.db[sequences_key]
                model = self.db[model_key]

                if distances.size(0) != batch.size(0):
                    logging.error(
                        f"Distance: {distances.size(0)}, Batch: ({batch.size()}), {batch_key} Sequence: {sequences_key}" + \
                        f"Model: {model_key}.")
                    continue
                yield (idx, type, batch, model, sequences, 
                        distances, rng_state, perturb_type)

        return (_batch_generator(self.train_queue, shuffle=shuffle), 
                    _batch_generator(self.valid_queue, shuffle=False))


def accumulate_scorer_training_data(step, batch_id, batch, buffer, model, 
                                      score_model, tokenizer, args, device):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The perturbations are sampled from $\Deta \sim Q_{MGS}$.

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """
    with timer_context('scorer_data_acc_time') as ctxt_timer:
      start_time = timer()
      batch.squeeze_(0)
      batch = batch.to(device=device)
      if batch.size(1) < args.context_length + 1:
          logging.error(
              f"Batch at step: {step} has sequences ({batch.size(1)})" + \
                f"shorter than the context length ({args.context_length})")
          return buffer

      inp, target = batch[:, :-1], batch[:, 1:]
      max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)
      model = model.to(device=device)
      model.eval()
      _, cur_decodings, cur_distances = ggs_utils.decode_and_distance(model, 
                                            tokenizer, batch, score_model, max_length, 
                                            device, args, average_distance=False)

      idx = f'accum_{step}'
      buffer.append(idx, InstanceType.NON_PERTURBED, batch_id, batch,
                     model, cur_decodings, cur_distances)

      # Get the current MLE gradients

      model_ = deepcopy(model)
      inp, target = batch[:, :-1], batch[:, 1:]
      model_with_grad, _ = ggs_utils.mle_grad(model_, 
                            inp, target, tokenizer.pad_token_id, 
                            args.max_grad_norm)

      perturbed_models, _, _, rng_states, perturb_types = \
            ggs_utils.perturb(model, model_with_grad, args.ggs_num_samples, 
                        args.ggs_noise, noise_scale=args.noise_scale,
                        zero_dist_only=args.zero_dist_only,
                        mle_dist_only=args.mle_dist_only,
                        include_mle_gradient=args.include_mle_gradient)

      for i, (p_model, rng_state, perturb_type) in \
            enumerate(zip(perturbed_models, rng_states, perturb_types)):
        _, per_decodings, per_distances = ggs_utils.decode_and_distance(p_model,
                                            tokenizer, batch, score_model, max_length,
                                            device, args, average_distance=False)
        # idx = f'accum_perturb_{step}_{i}'
        idx = f'accum_{step}'
        buffer.append(idx, InstanceType.PERTURBED, batch_id, batch, p_model, 
                        per_decodings, per_distances, rng_state=rng_state,
                        perturb_type=perturb_type)

      end_time = timer()
      avg_time = ctxt_timer.timeit(start_time, end_time)
    return buffer


def perturb(model, batch, step, tokenizer, args, device=None,
                        rng_state=None,  apply_to_mle_grad=None):

    apply_to_mle_grad = apply_to_mle_grad or (random.random() < 0.5)

    per_model = deepcopy(model)
    inp, target = batch[:, :-1], batch[:, 1:]

    model_with_grad, _ = ggs_utils.mle_grad(per_model, inp, target, 
                              tokenizer.pad_token_id, args.max_grad_norm)

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


def get_train_score_network_loss(idx, type, model, batch, distances, sequences,
                                      score_network, tokenizer, device, args):
    model = model.to(device=device)
    model.eval()

    batch = batch.to(device=device)

    outputs = score_network(model, batch, predictions=sequences)

    distances = distances.to(device=device)
    if distances.size(0) != outputs.size(0):
        logging.error(f"Batch: ({idx} {type} {batch.size()}) != Distance {distances.size()}")
        return -1.0

    losses = F.mse_loss(
        outputs,
        distances.view(-1, 1),
        reduction='none',
    )
    return losses, outputs

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

        self.num_docs = 0
        self.num_docs_pert = 0
        self.num_docs_non_pert = 0

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
        self.disagreement_count = 0
        
        self.cuml_pred_dist = 0.
        self.cuml_true_dist = 0.
        
        self.cuml_pred_dist_pert = 0.
        self.cuml_pred_dist_non_pert = 0.
        self.cuml_true_dist_pert = 0.
        self.cuml_true_dist_non_pert = 0.

        self.prefix = prefix

    def __call__(self, idx, type, batch, 
                 losses, true_dist, pred_dist):
        losses = losses.squeeze(1)
        
        self.cuml_loss += losses.sum().item()

        self.true_dist += true_dist.tolist()
        self.pred_dist += pred_dist.tolist()
        self.loss_list += losses.tolist()

        scorer_fit = (torch.abs((true_dist - pred_dist)/true_dist))\
                        .sum().item()
        
        self.cuml_scorer_fit_all += scorer_fit
        self.num_docs += batch.size(0)

        if idx not in self.scorer_diff_fit_dict:
            self.scorer_diff_fit_dict[idx] = {
                                   'non_pert': None, 
                                   'perturbed': [],
                                }
        
        self.cuml_pred_dist += pred_dist.sum().item()
        self.cuml_true_dist += true_dist.sum().item()
        if type == InstanceType.PERTURBED:
            self.num_docs_pert += batch.size(0)
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
            self.num_docs_non_pert += batch.size(0)
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

        self.num_docs_pert = 0
        self.num_docs_non_pert = 0

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

                self.disagreements += ((perturb_pred_score - non_pert_pred_score)/
                                        (perturb_true_score - non_pert_true_score) < 0).item()
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

          f"{self.prefix}/true_dist_mean": self.cuml_true_dist/self.num_docs, 
          f"{self.prefix}/true_dist_pert_mean": self.cuml_true_dist_pert/self.num_docs_pert, 
          f"{self.prefix}/true_dist_non_pert_mean": self.cuml_true_dist_non_pert/self.num_docs_non_pert,

          f"{self.prefix}/pred_dist_mean":self.cuml_pred_dist/self.num_docs, 
          f"{self.prefix}/pred_dist_pert_mean":self.cuml_pred_dist_pert/self.num_docs_pert,
          f"{self.prefix}/pred_dist_non_pert_mean":self.cuml_pred_dist_non_pert/self.num_docs_non_pert, 
        }

def validate_score_network(valid_iter, score_network, tokenizer, device, args):
    cuml_valid_loss = 0.
    num_docs = 0
    valid_scorer_analysis = ScorerAnalysis('valid')
    score_network.eval()
    for step, (idx, type, batch, model, sequences, distances,
                rng_state, perturb_type) in enumerate(valid_iter):

        if type == InstanceType.PERTURBED:
            model_ = deepcopy(model)
            inp, target = batch[:, :-1], batch[:, 1:]
            model_with_grad, _ = ggs_utils.mle_grad(model_, inp, target, 
                                    tokenizer.pad_token_id, args.max_grad_norm)

            model, _, _, _ = ggs_utils.perturb_single(model, model_with_grad, 
                                args.ggs_noise, noise_scale=args.noise_scale,
                                perturb_type=perturb_type, rng_state=rng_state)

        losses, pred_distances = get_train_score_network_loss(idx, 
                                        type, model, batch, distances, sequences,
                                        score_network, tokenizer, device, args)

        batch_loss = torch.sqrt(losses).sum().item()
        if batch_loss < 0:
            continue

        cuml_valid_loss += batch_loss
        num_docs += batch.size(0)
        pred_distances = pred_distances.squeeze(1).detach()
        valid_scorer_analysis(idx, type, batch, torch.sqrt(losses), distances, pred_distances)

        if step % 5 == 0 and step > 0:
            print('Validation:: Step: %d, Loss: %.3f'
                  % (step, cuml_valid_loss / num_docs), end='\r')
    print()

    valid_info_dict = valid_scorer_analysis.get_metrics()
    return cuml_valid_loss/num_docs, valid_info_dict


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

    score_network = score_network.to(device=device)

    phi_optimizer = optim.AdamW(score_network.parameters(), lr=args.scorer_lr)
    scheduler = optim.lr_scheduler.StepLR(phi_optimizer, step_size=20, gamma=0.1, verbose=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(phi_optimizer, 'min', patience=3, verbose=True)

    # _, valid_iterator = buffers.get_iterators()
    # valid_loss, valid_info_dict = validate_score_network(
    #                                 valid_iterator, score_network,
    #                                 tokenizer, device, args)

    for epoch in range(epochs):
        score_network.train()
        train_scorer_analysis = ScorerAnalysis('train')
        train_iterator, valid_iterator = buffers.get_iterators()
        cuml_train_loss = 0.
        num_docs = 0
        
        with timer_context('train_score_network_time') as ctxt_timer:
            start_time = timer()
            for step, (idx, type, batch, model, sequences, distances,
                         rng_state, perturb_type) in enumerate(train_iterator):

                phi_optimizer.zero_grad()
                if type == InstanceType.PERTURBED:
                    inp, target = batch[:, :-1], batch[:, 1:]
                    model_ = deepcopy(model)
                    model_with_grad, _ = ggs_utils.mle_grad(model_, inp, target, 
                                            tokenizer.pad_token_id, args.max_grad_norm)

                    model, _, _, _ = ggs_utils.perturb_single(model, 
                                        model_with_grad, 
                                        args.ggs_noise, 
                                        noise_scale=args.noise_scale,
                                        perturb_type=perturb_type, 
                                        rng_state=rng_state)
        
                losses, pred_distances = get_train_score_network_loss(idx, 
                                                type, model, batch, distances, sequences,
                                                score_network, tokenizer, device, args)

                loss = losses.sum()/batch.size(0)

                if loss < 0:
                    continue

                pred_distances = pred_distances.squeeze(1).detach()
                train_scorer_analysis(idx, type, batch, torch.sqrt(losses), distances, pred_distances)

                cuml_train_loss += torch.sqrt(losses).sum().item()
                num_docs += batch.size(0)

                loss.backward()
                phi_optimizer.step()

                if step % 5 == 0 and step > 0:
                    print('Training:: Epoch: %d :: Step: %d, Loss: %.3f' % (epoch, step, cuml_train_loss / num_docs),
                        end='\r')

                if not args.on_device:
                    # Move model back to CPU so that it doesn't hog GPU
                    # memory as it will not be removed from the context.
                    model.to(device=torch.device("cpu"))
                    distances.to(device=torch.device("cpu"))

            print()
            end_time = timer()
            train_info_dict = train_scorer_analysis.get_metrics()
            print(pformat(train_info_dict))

            ctxt_timer.timeit(start_time, end_time)
        
        with timer_context('validate_score_network_time') as ctxt_timer:
            start_time = timer()
            train_loss = cuml_train_loss / num_docs
            valid_loss, valid_info_dict = validate_score_network(
                                            valid_iterator, score_network,
                                            tokenizer, device, args)
            # scheduler.step(valid_loss)
            scheduler.step()

            if min_valid_loss < valid_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                min_valid_loss = valid_loss
                best_score_network = deepcopy(score_network)

            print(pformat(valid_info_dict))
            if patience_counter > args.train_score_patience:
                logging.info(f"Stopping Early at epoch: {epoch} with best validation loss: {min_valid_loss}")
                break

            end_time = timer()
            ctxt_timer.timeit(start_time, end_time)

        logging.info('Epoch: %d :: Train Loss: %.3f, ' % (epoch, train_loss) +
                     'Best Valid Loss: %.3f, Valid Loss: %.3f, Epochs Since Last Best: %d '
                     % (min_valid_loss, valid_loss, patience_counter))
        logging.info(f"Train score network epoch {epoch} done!")
        logging.info(f"Avg Epoch Time: " +
                     f"Train: {timer_context('train_score_network_time').avg_time():.3f} " +
                     f"Valid: {timer_context('validate_score_network_time').avg_time():.3f}.")

        prefix = f"tsn-{train_score_network_iteration}/"
        valid_metrics = {
            prefix + "train_loss": train_loss,
            prefix + "valid_loss": valid_loss,
            prefix + "min_valid_loss": min_valid_loss,
            prefix + "epochs_since_best": patience_counter,
            prefix + "current_epoch": epoch,
        }

        for key, val in valid_info_dict.items():
            valid_metrics[prefix + key] = val

        for key, val in train_info_dict.items():
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
      pad = tokenizer.pad_token_id
      batch = batch.to(device=model.device)

      batched_distances = score_network(model, batch) \
                              .detach().cpu()

      # average across batch to compute c(\theta).
      distances = batched_distances.mean(dim=0).item()
      
      end_time = timer()
      ctxt_timer.timeit(start_time, end_time)
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
        "--scorer-lr", type=float, default=5e-4,
    )

    parser.add_argument(
        "--initialize-score-network", action='store_true',
    )
    score_network_utils.add_args(parser)
    return parser