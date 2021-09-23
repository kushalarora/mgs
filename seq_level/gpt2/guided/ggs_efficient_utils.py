from collections import defaultdict
from copy import deepcopy
from pprint import pformat
from scipy.stats import kendalltau
from timeit import default_timer as timer


import hashlib
import logging
import math
import os
import random
import shelve
import torch
import torch.nn.functional as F
import torch.optim as optim

import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils

timer_context = ggs_utils.TimerContext()
def _hash_tensor(obj):
    return hashlib.sha1(bytes(obj.cpu().numpy())).hexdigest()


def _hash_model(model):
    return hashlib.sha1(next(model.parameters()).detach().cpu().numpy()).hexdigest()

MODEL_ID = None

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

        # global MODEL_ID
        model_key = f"model_{MODEL_ID}"
        # model_key = f"model_{_hash_model(model)}"

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

            for (idx, type, batch_key, model_key, sequences_key, 
                    distances, rng_state, apply_to_mle_grad) in iterable:

                batch = self.db[batch_key].type(torch.long)
                sequences = None
                model = self.db[model_key]

                if distances.size(0) != batch.size(0):
                    logging.error(
                        f"Distance: {distances.size(0)}, Batch: ({batch.size()}), {batch_key} Sequence: {sequences_key}" + \
                        f"Model: {model_key}.")
                    continue
                yield (idx, type, batch, model, sequences, 
                        distances, rng_state, apply_to_mle_grad)

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
      buffer.append(idx, 'current', batch_id, batch,
                     model, cur_decodings, cur_distances)

      # Get the current MLE gradients
      model.train()
      per_model, rng_state, apply_to_mle_grad = perturb(model, batch, 
                                                        step, tokenizer, 
                                                        args, device=device)

      _, per_decodings, per_distances = ggs_utils.decode_and_distance(per_model,
                                          tokenizer, batch, score_model, max_length,
                                          device, args, average_distance=False)

      buffer.append(idx, 'perturbed', batch_id, batch, model, 
                      per_decodings, per_distances, 
                      rng_state=rng_state, 
                      apply_to_mle_grad=apply_to_mle_grad)

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


def get_train_score_network_loss(idx, type, model, batch, distances, 
                                      score_network, tokenizer, device):
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
    cuml_perturbed_loss = 0.
    cuml_non_perturbed_loss = 0.

    cuml_scorer_fit_all = 0.
    cuml_scorer_fit_perturbed = 0.
    cuml_scorer_fit_original = 0.

    num_docs = 0
    num_docs_perturbed = 0
    num_docs_non_perturbed = 0

    # idx => {'original': (true_dist, pred_dist), 'perturbed': [(true_dist, pred_dist)]}
    scorer_diff_fit_dict = {}

    true_distances = []
    predicted_distances = []

    true_distances_perturbed = []
    true_distances_non_perturbed = []
    predicted_distances_perturbed = []
    predicted_distances_non_perturbed = []

    for step, (idx, type, batch, model, _, distances,
                rng_state, apply_to_mle_grad) in enumerate(valid_iter):

        if type == "perturbed":
            model, _, _ = perturb(model, batch, idx, tokenizer, args, 
                                  device=device, rng_state=rng_state,
                                  apply_to_mle_grad=apply_to_mle_grad)

        loss, pred_distances = get_train_score_network_loss(idx, 
                                    type, model, batch, distances, 
                                    score_network, tokenizer, device)

        if loss < 0:
            continue

        pred_distances = pred_distances.squeeze(1).detach()

        true_distances += distances.tolist()
        predicted_distances += pred_distances.tolist()
        scorer_fit = torch.sum(torch.abs(distances - pred_distances)/pred_distances).item()
        
        cuml_valid_loss += loss.item()
        cuml_scorer_fit_all += scorer_fit

        num_docs += batch.size(0)

        if idx not in scorer_diff_fit_dict:
            scorer_diff_fit_dict[idx] = {'original': None, 
                                          'perturbed': [],}
        
        if type == "perturbed":
            num_docs_perturbed += batch.size(0)
            cuml_perturbed_loss += loss.item()
            true_distances_perturbed += distances.tolist()
            predicted_distances_perturbed += pred_distances.tolist()
            cuml_scorer_fit_perturbed = scorer_fit
            scorer_diff_fit_dict[idx]['perturbed'].append((distances.mean(), pred_distances.mean()))
        else:
            num_docs_non_perturbed += batch.size(0)
            cuml_non_perturbed_loss += loss.item()
            true_distances_non_perturbed += distances.tolist()
            predicted_distances_non_perturbed += pred_distances.tolist()
            cuml_scorer_fit_original += scorer_fit
            scorer_diff_fit_dict[idx]['original'] = (distances.mean(), pred_distances.mean())

        if step % 5 == 0 and step > 0:
            print('Validation:: Step: %d, Loss: %.3f'
                  % (step, cuml_valid_loss / num_docs), end='\r')
    print()

    cuml_scorer_diff_fit = 0.
    func_diff_fit_count = 0
    for idx, diff_dict in scorer_diff_fit_dict.items():
        original_true_score, original_pred_score = diff_dict['original']
        pertured_scores = diff_dict['perturbed']

        for perturb_true_score, perturb_pred_score in pertured_scores:
            if (perturb_true_score - original_true_score) == 0:
                logging.debug(f"For idx: {idx}," + 
                                f" perturbed score: {perturb_true_score:.3f}" + 
                                f" original score: {original_true_score:.3f}" + 
                                " are equal.")
                continue

            cuml_scorer_diff_fit += torch.abs((perturb_pred_score - original_pred_score)/(perturb_true_score - original_true_score)).item()
            func_diff_fit_count += 1

    valid_info_dict = {
          "all_corr": kendalltau(true_distances, predicted_distances)[0],
          "perturbed_corr": kendalltau(true_distances_perturbed, 
                                        predicted_distances_perturbed)[0],
          "original_corr": kendalltau(true_distances_non_perturbed,
                                        predicted_distances_non_perturbed)[0],
          "perturbed_loss": cuml_perturbed_loss / num_docs_perturbed,
          "original_loss": cuml_non_perturbed_loss / num_docs_non_perturbed, 
          "scorer_fit_all": cuml_scorer_fit_all / num_docs,
          "scorer_fit_perturbed": cuml_scorer_fit_perturbed / num_docs_perturbed,
          "scorer_fit_original": cuml_scorer_fit_original / num_docs_non_perturbed,
          "scorer_diff_fit": cuml_scorer_diff_fit / func_diff_fit_count,
        }
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

    score_network.train()
    score_network = score_network.to(device=device)

    phi_optimizer = optim.Adam(score_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(phi_optimizer, step_size=20, gamma=0.5, verbose=True)

    for epoch in range(epochs):
        train_iterator, valid_iterator = buffers.get_iterators()
        cuml_train_loss = 0.
        num_docs = 0
        
        with timer_context('train_score_network_time') as ctxt_timer:
            start_time = timer()
            for step, (idx, type, batch, model, _, distances, rng_state,
                            apply_to_mle_grad) in enumerate(train_iterator):

                phi_optimizer.zero_grad()
                if type == "perturbed":
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
                    print('Training:: Epoch: %d :: Step: %d, Loss: %.3f' % (epoch, step, cuml_train_loss / num_docs),
                        end='\r')

                if not args.on_device:
                    # Move model back to CPU so that it doesn't hog GPU
                    # memory as it will not be removed from the context.
                    model.to(device=torch.device("cpu"))
                    distances.to(device=torch.device("cpu"))

            print()
            end_time = timer()
            ctxt_timer.timeit(start_time, end_time)
        
        with timer_context('validate_score_network_time') as ctxt_time:
            start_time = timer()
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

            if not args.only_train_score_network and \
                patience_counter > args.train_score_patience:
                logging.info(f"Stopping Early at epoch: {epoch} with best validation loss: {min_valid_loss}")
                break

            scheduler.step()
            
            end_time = timer()
            ctxt_timer.timeit(start_time, end_time)

        logging.info('Epoch: %d :: Train Loss: %.3f, ' % (epoch, train_loss) +
                     'Best Valid Loss: %.3f, Valid Loss: %.3f, Epochs Since Last Best: %d '
                     % (min_valid_loss, valid_loss, patience_counter))
        logging.info(f"Train score network epoch {epoch} done!")
        logging.info(f"Avg Epoch Time: " +
                     f"Train: {timer_context('train_score_network_time').avg_time():.3f} " +
                     f"Valid: {timer_context('validate_score_network_time').avg_time():.3f}.")

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

      batched_distances = score_network(model, batch, pad) \
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

