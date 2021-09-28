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

import seq_level.gpt2.guided.score_network as score_network_utils
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
                        apply_to_mle_grad))

    def get_iterators(self, shuffle=True):
        def _batch_generator(iterable, shuffle=True):
            if shuffle:
                iterable = random.sample(iterable, len(iterable))

            for (idx, type, batch_key, model_key, sequences_key, 
                    distances, rng_state, apply_to_mle_grad) in iterable:

                batch = self.db[batch_key]
                sequences = self.db[sequences_key]
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

    loss = F.mse_loss(
        outputs,
        distances.view(-1, 1),
        reduction='sum',
    )
    return loss, outputs

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

        self.true_dist = []
        self.pred_dist = []

        self.true_dist_pert = []
        self.true_dist_non_pert = []
        self.pred_dist_pert = []
        self.pred_dist_non_pert = []

        self.cuml_scorer_diff_fit = 0.
        self.scorer_diff_fit_count = 0
        
        self.cuml_pred_dist = 0.
        self.cuml_true_dist = 0.
        
        self.cuml_pred_dist_pert = 0.
        self.cuml_pred_dist_non_pert = 0.
        self.cuml_true_dist_pert = 0.
        self.cuml_true_dist_non_pert = 0.

        self.prefix = prefix

    def __call__(self, idx, type, batch, 
                 loss, true_dist, pred_dist):
        self.cuml_loss += loss.item()

        self.true_dist += true_dist.tolist()
        self.pred_dist += pred_dist.tolist()
        
        scorer_fit = (torch.abs(true_dist - pred_dist)/pred_dist)\
                        .sum().item()
        
        self.cuml_scorer_fit_all += scorer_fit
        self.num_docs += batch.size(0)

        if idx not in self.scorer_diff_fit_dict:
            self.scorer_diff_fit_dict[idx] = {
                                   'non_pert': None, 
                                   'perturbed': [],
                                }
        
        self.cuml_pred_dist += pred_dist.mean().item()
        self.cuml_true_dist += true_dist.mean().item()
        if type == "perturbed":
            self.num_docs_pert += batch.size(0)
            self.cuml_pert_loss += loss.item()
            self.true_dist_pert += true_dist.tolist()
            self.pred_dist_pert += pred_dist.tolist()
            self.cuml_scorer_fit_pert += scorer_fit
            self.scorer_diff_fit_dict[idx]['perturbed'].append((true_dist.mean(), 
                                                                pred_dist.mean()))
            self.cuml_pred_dist_pert += pred_dist.mean().item()
            self.cuml_true_dist_pert += true_dist.mean().item()
        else:
            self.num_docs_non_pert += batch.size(0)
            self.cuml_non_pert_loss += loss.item()
            self.true_dist_non_pert += true_dist.tolist()
            self.pred_dist_non_pert += pred_dist.tolist()
            self.cuml_scorer_fit_non_pert += scorer_fit
            self.scorer_diff_fit_dict[idx]['non_pert'] = (true_dist.mean(), 
                                                          pred_dist.mean())
            self.cuml_pred_dist_non_pert += pred_dist.mean().item()
            self.cuml_true_dist_non_pert += true_dist.mean().item()

    def reset(self):
        self.scorer_diff_fit_dict = {}
        self.cuml_pert_loss = 0.
        self.cuml_non_pert_loss = 0.

        self.cuml_scorer_fit_all = 0.
        self.cuml_scorer_fit_pert = 0.
        self.cuml_scorer_fit_non_pert = 0.

        self.num_docs_pert = 0
        self.num_docs_non_pert = 0


        self.true_dist = []
        self.pred_dist = []

        self.true_dist_pert = []
        self.true_dist_non_pert = []
        self.pred_dist_pert = []
        self.pred_dist_non_pert = []
        
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

                self.cuml_scorer_diff_fit += torch.abs((perturb_pred_score - non_pert_pred_score)/(perturb_true_score - non_pert_true_score)).item()
                self.scorer_diff_fit_count += 1

        return {
          f"{self.prefix}/corr_all": kendalltau(self.true_dist, self.pred_dist)[0],
          f"{self.prefix}/corr_pert": kendalltau(self.true_dist_pert, self.pred_dist_pert)[0],
          f"{self.prefix}/corr_non_pert": kendalltau(self.true_dist_non_pert,
                                                     self.pred_dist_non_pert)[0],

          f"{self.prefix}/loss_all": self.cuml_loss / self.num_docs,
          f"{self.prefix}/loss_pert": self.cuml_pert_loss / self.num_docs_pert,
          f"{self.prefix}/loss_non_pert": self.cuml_non_pert_loss / self.num_docs_non_pert, 

          f"{self.prefix}/scorer_fit_all": self.cuml_scorer_fit_all / self.num_docs,
          f"{self.prefix}/scorer_fit_pert": self.cuml_scorer_fit_pert / self.num_docs_pert,
          f"{self.prefix}/scorer_fit_non_pert": self.cuml_scorer_fit_non_pert / self.num_docs_non_pert,

          f"{self.prefix}/scorer_diff_fit": self.cuml_scorer_diff_fit / self.scorer_diff_fit_count,

          f"{self.prefix}/avg_true_dist_all": self.cuml_true_dist/self.num_docs, 
          f"{self.prefix}/avg_true_dist_pert": self.cuml_true_dist_pert/self.num_docs_pert, 
          f"{self.prefix}/avg_true_dist_non_pert": self.cuml_true_dist_non_pert/self.num_docs_non_pert,

          f"{self.prefix}/avg_pred_dist_all":self.cuml_pred_dist/self.num_docs, 
          f"{self.prefix}/avg_pred_dist_pert":self.cuml_pred_dist_pert/self.num_docs_pert,
          f"{self.prefix}/avg_pred_dist_non_pert":self.cuml_pred_dist_non_pert/self.num_docs_non_pert, 
        }

def validate_score_network(valid_iter, score_network, tokenizer, device, args):
    cuml_valid_loss = 0.
    num_docs = 0
    valid_scorer_analysis = ScorerAnalysis('valid')
    score_network.eval()
    for step, (idx, type, batch, model, sequences, distances,
                rng_state, apply_to_mle_grad) in enumerate(valid_iter):

        if type == "perturbed":
            model, _, _ = perturb(model, batch, idx, tokenizer, args, 
                                  device=device, rng_state=rng_state,
                                  apply_to_mle_grad=apply_to_mle_grad)

        loss, pred_distances = get_train_score_network_loss(idx, 
                                    type, model, batch, distances, sequences,
                                    score_network, tokenizer, device, args)

        if loss < 0:
            continue

        cuml_valid_loss += loss.item()
        num_docs += batch.size(0)
        pred_distances = pred_distances.squeeze(1).detach()
        valid_scorer_analysis(idx, type, batch, loss, distances, pred_distances)

        if step % 5 == 0 and step > 0:
            print('Validation:: Step: %d, Loss: %.3f'
                  % (step, cuml_valid_loss / num_docs), end='\r')
    print()

    valid_info_dict = valid_scorer_analysis.get_metrics()
    return cuml_valid_loss/max(num_docs, 1), valid_info_dict


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
    scheduler = optim.lr_scheduler.StepLR(phi_optimizer, step_size=10, gamma=0.5, verbose=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(phi_optimizer, 'min', patience=5, verbose=True)

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
            for step, (idx, type, batch, model, sequences, distances, rng_state,
                            apply_to_mle_grad) in enumerate(train_iterator):

                phi_optimizer.zero_grad()
                if type == "perturbed":
                    model, _, _ = perturb(model, batch, idx, tokenizer, args, 
                                    device=device, rng_state=rng_state, apply_to_mle_grad=apply_to_mle_grad)
        
                loss, pred_distances = get_train_score_network_loss(idx, 
                                            type, model, batch, distances, sequences,
                                            score_network, tokenizer, device, args)

                if loss < 0:
                    continue

                pred_distances = pred_distances.squeeze(1).detach()
                train_scorer_analysis(idx, type, batch, loss, distances, pred_distances)

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

        prefix = f"train_score_network_{train_score_network_iteration}/"
        valid_metrics = {
            prefix + "valid_loss": valid_loss,
            prefix + "min_valid_loss": min_valid_loss,
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

    parser.add_argument('--use-learned-scoring-function', 
                        action='store_true')

    parser.add_argument('--only-train-score-network', 
                        action='store_true')

    parser.add_argument(
        "--train-score-patience", type=int, default=20,
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
        "--scorer-lr", type=float, default=2e-4,
    )

    score_network_utils.add_args(parser)
    return parser