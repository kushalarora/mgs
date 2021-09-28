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
import seq_level.gpt2.guided.ggs_efficient_utils as dagger_ggs_utils

import os
from functools import partial
from seq_level.gpt2.guided.metrics import GuidedMetrics
from seq_level.gpt2.guided.score_network import ScoreNetwork
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

from timeit import default_timer as timer
import pandas as pd
import collections
import random
import pickle
import logging
import scipy.stats as stats


timer_context = ggs_utils.TimerContext()

def original_mgs_scoring_function(buffer, is_target_function, model, 
        tokenizer, batch, score_model, max_length, device,  args, prefix):
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


def MGS(batch, model, score_model, tokenizer, args, device, metrics, optimizer,
        scoring_function=None,
        target_scoring_func=None):
    """
    MGS algorithm parameterized to work in original as well as efficient mode.
    """
    distance_comp = []
    inp, target = batch[:, :-1].to(device=device), batch[:, 1:].to(device=device)

    # -- Decode with current model (required for computing the 'weights' later).
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

    decoded = defaultdict(list)

    with timer_context('total_scoring_time') as ctxt_timer:
        start_time = timer()
        distance_curr, bpes_curr, decoded_samples = scoring_function(
            model, tokenizer, batch, score_model, max_length, device, args, prefix='original'
        )

        if args.efficient and args.log_scoring_function and args.log_step % args.print_every == 1:
            distance_curr_score, _, _ = target_scoring_func(
                model, tokenizer, batch, score_model, max_length, device, args, prefix='original'
            )
            distance_comp.append(('original', distance_curr, distance_curr_score))
            logging.info(f"Distances: original: C => {distance_curr:.3f} C_t => {distance_curr_score:.3f}")

        decoded.update(decoded_samples)

        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    # -- Obtain MLE gradients
    with timer_context('mle_grad_computation_time') as ctxt_timer:
        start_time = timer()
        model_ = deepcopy(model)
        model_with_grad, mle_loss = ggs_utils.mle_grad(
            model_, inp, target, tokenizer.pad_token_id, args.max_grad_norm
        )
        
        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    # -- Perturb
    with timer_context('perturb_computation_time') as ctxt_timer:
        start_time = timer()
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
        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    with timer_context('perturb_scoring_time') as perturb_scoring_timer, \
        timer_context('total_scoring_time') as total_scoring_timer:
        # -- Decode with perturbed models and compute task metric
        start_time = timer()
        distances = []
        for i, p_model in enumerate(perturbed_models):

            distance, _, decoded_samples = scoring_function(p_model, tokenizer, batch, score_model,
                                                            max_length, device, args, prefix=f'perturb_{i}'
                                                            )
            if args.efficient and args.log_scoring_function and args.log_step % args.print_every == 1:
                distance_score, _, _ = target_scoring_func(p_model, tokenizer, batch,
                                                        score_model, max_length, device, args, prefix='perturb_{i}')
                distance_comp.append(('perturb_{i}', distance, distance_score))
                logging.info(f"Distances: perturb_{i}: C => {distance:.3f} C_t => {distance_score:.3f}")
                logging.info(
                    f"C'_{i} - C: => {(distance - distance_curr):.3f} C'_t_{i} - C_t => {(distance_score - distance_curr_score):.3f}")

            distances.append(distance)
            decoded.update(decoded_samples)
        end_time = timer()
        perturb_scoring_timer.timeit(start_time, end_time)
        total_scoring_timer.timeit(start_time, end_time)


    # -- Compute weights
    # Kushal: please revise score_network(distance_curr - distances), where the scorer's output is embedding.

    with timer_context('weight_computation_time') as ctxt_timer:
        start_time = timer()
        log_weights = ggs_utils.compute_weight(distance_curr, distances, log_rhos, args.ggs_beta)

        # -- Compute weighted average of the directions
        update_directions = ggs_utils.parameter_weighted_average(
            model, perturbed_models, log_weights
        )
        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    with timer_context('ggs_update_time') as ctxt_timer:
        start_time = timer()
        # -- Perform update
        # global ggs_utils.MODEL_ID
        ggs_utils.MODEL_ID = ggs_utils.update(model, update_directions, 
                                    optimizer, args.max_grad_norm)

        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    metrics_update_start = timer()
    with timer_context('metrics_update_time') as ctxt_timer:
        start_time = timer()
        # -- Record statistics
        metrics.step(
            mle_loss.item(), distance_curr, bpes_curr, target, args.context_length,
            tokenizer.pad_token_id, tokenizer.eos_token_id,
            model_with_grad, update_directions, log_rhos, log_weights,
            noise_magnitudes, distances
        )
        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)
    return decoded

def train(model, tokenizer, dataset_tensor_dict, args, device):
    # global ggs_utils.MODEL_ID
    ggs_utils.MODEL_ID = ggs_utils.get_model_id(model)

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

    if args.efficient:
        # Initialize buffer and pretrain score network.
        print('=' * 100)

        # If using saved aggregated data, use it, else, initialize an empty buffer.
        if args.use_saved_aggregated_data:
            with open(args.aggregated_data_path, 'rb') as aggregated_datafile:
                buffer = pickle.load(aggregated_datafile)
            logging.info(f"Loading Aggregated data from {args.aggregated_data_path}." + \
                            f" Size: {len(buffer)}")
        else:
            buffer = dagger_ggs_utils.RingBuffer(max_size=args.max_buffer_size,
                                        persistence='none',
                                        persistent_file_path=os.path.join(
                                            args.save_base_dir, "persistence_datastore"),
                                        on_device=args.on_device)

        initial_training_epochs = args.score_network_epochs
        # If using saved score network, use it, else accumulate training data, 
        # and train network on the accumulated data.
        if args.use_saved_score_network:
            checkpoint = torch.load(args.score_network_file)
            score_network.load_state_dict(checkpoint['model_save_dict'])
            epochs = score_model_checkpoint['epochs']
            logging.info(f"Loading scorer trained for {epochs} epochs" + \
                            f" from {args.score_network_file}.")
            initial_training_epochs = args.retrain_score_network_epochs
        else:
            score_network = ScoreNetwork(input_size=config.hidden_size) \
                                .to(device=device)

            if not args.use_saved_aggregated_data:
                logging.info("Started Initial Data Accumulation.")

                for step, (batch_id, batch) in enumerate(train_dataloader):
                    if step >= args.aggregated_data_size:
                        break

                    dagger_ggs_utils.accumulate_scorer_training_data(step,
                                            batch_id, batch, buffer, model, 
                                            score_model, tokenizer, args, device)


                    
                    scorer_acc_timer = timer_context.get_timer('scorer_data_acc_time')
                    if step % args.print_every == 0:
                        logging.info(f"Aggregated Batches:  {step}/{total_num_batches}. " + \
                                     f"Avg step time: {scorer_acc_timer.avg_time():.3f}")

                scorer_acc_timer = timer_context.get_timer('scorer_data_acc_time')
                logging.info(f"Aggregated: {len(buffer)} items in " + \
                             f"{scorer_acc_timer.cuml_time():.3f} seconds.")

                if args.save_aggregated_data:
                    buffer_filepath = os.path.join(args.save_base_dir, 'buffer.pkl')
                    logging.info(f"Saving Aggregated Data at {buffer_filepath}")
                    with open(buffer_filepath, 'wb') as buffer_file:
                        pickle.dump(buffer, buffer_file)

        logging.info("Training Scoring Network on Aggregated Data.")
        dagger_ggs_utils.train_score_network(buffer, score_network, tokenizer,
                                        device, args, score_network_training_iter, 
                                        epochs=initial_training_epochs)

        scoring_function = partial(original_mgs_scoring_function, buffer, False)
        target_scoring_func = partial(dagger_ggs_utils.dagger_mgs_scoring_function, score_network)

        if args.use_learned_scoring_function:
            scoring_function = partial(dagger_ggs_utils.dagger_mgs_scoring_function, score_network)
            target_scoring_func = partial(original_mgs_scoring_function, buffer, False)
        print('=' * 100)
    
    if args.only_train_score_network:
        return

    for epoch_number in range(args.num_train_epochs):
        metrics = GuidedMetrics()

        for step, (batch_id, batch) in enumerate(train_dataloader):
            with timer_context('total_step_time') as total_ctxt_timer:
                train_step_start = timer()
                if args.efficient:
                    if dagger_ggs_utils.shall_accumulate_scorer_training_data(step, total_num_batches, args):
                        dagger_ggs_utils.accumulate_scorer_training_data(step, batch_id, 
                                                        batch, buffer, model, score_model, 
                                                        tokenizer, args, device)

                    if (step + 1) % args.retrain_score_network_every == 0:
                        score_network_training_iter += 1
                        dagger_ggs_utils.train_score_network(buffer, score_network,
                            tokenizer, device, args, score_network_training_iter,
                            epochs=args.retrain_score_network_epochs,)

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

                with timer_context('total_mgs_time') as ctxt_timer:
                    mgs_start_time = timer()
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
                    mgs_end_time = timer()
                    ctxt_timer.timeit(mgs_start_time, mgs_end_time)

                train_step_end = timer()
                total_ctxt_timer.timeit(train_step_start, train_step_end)

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

                average_times = timer_context.get_avg_times()
                # if args.plot_times:
                #     df = pd.DataFrame.from_dict(average_times,
                #                                 orient='index',
                #                                 columns=['avg. time'])
                    # print(df)
                utils.log_tensorboard(average_times, 0)

                if not args.use_learned_scoring_function and args.print_decodings:
                    i = random.choice(range(batch.size(0)))
                    print(f"Step: {step}")
                    print(f"theta(x_{i}): {decoded['original_%d' % i][0]}")
                    for j in range(args.ggs_num_samples):
                        print(f"theta+Delta_{j}(x_{i}): {decoded['perturb_%d_%d' % (j, i)][0]}")
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
                        print(f"{'Prefix':10}: {prefixes[i]}")
                        print(f"{'Sequence':10}: {sentences[i]}")                    
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

    parser = dagger_ggs_utils.add_args(parser)

    return parser
