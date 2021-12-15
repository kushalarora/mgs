import torch
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import GPT2Config
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils
import seq_level.gpt2.train as train_utils
import seq_level.gpt2.guided.dagger_ggs.ggs_efficient_utils as dagger_ggs_utils

import os
from functools import partial
from seq_level.gpt2.guided.metrics import GuidedMetrics
from seq_level.gpt2.guided.dagger_ggs.ggs_efficient_utils import InstanceType
import seq_level.gpt2.guided.dagger_ggs.score_network as score_network_utils

import seq_level.gpt2.guided.dagger_ggs.score_network_trainer as score_network_trainer
import seq_level.gpt2.guided.dagger_ggs.accumulator as accumulator

from pprint import pformat

from pytorch_lightning import (LightningModule, 
                               Trainer)

from pytorch_lightning.plugins import DDPSpawnPlugin

from pytorch_lightning.loggers import (TensorBoardLogger, 
                                        WandbLogger)

from timeit import default_timer as timer
import pandas as pd
import collections
import random
import pickle
import logging
import scipy.stats as stats


timer_context = ggs_utils.TimerContext()

def original_mgs_scoring_function(model, 
        tokenizer, batch, score_model, max_length, device,  args, prefix):
    is_original = True
    decoded = defaultdict(list)
    bpes_curr, outputs, distance_curr = ggs_utils.decode_and_distance(
        model, tokenizer, batch, score_model,
        max_length, device, args, average_distance=False)

    if not isinstance(batch, list):
        output_list = outputs.tolist()

    for i, decoding in enumerate(output_list):
        if tokenizer.eos_token_id in decoding:
            decoding = decoding[:decoding.index(tokenizer.eos_token_id) + 1]
        decoded[f'{prefix}_{i}'] = [tokenizer.decode(decoding), distance_curr[i]]
    return is_original, distance_curr, bpes_curr, outputs, decoded


def MGS(batch_id, batch, model, score_model, tokenizer, args, device, metrics, 
        optimizer, scoring_function=None, target_scoring_func=None, 
        use_learned_scoring_func=False, buffer=None):
    """
    MGS algorithm parameterized to work in original as well as efficient mode.
    """
    inp, target = batch[:, :-1], batch[:, 1:]

    # -- Decode with current model (required for computing the 'weights' later).
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

    distance_comp = []
    decoded = defaultdict(tuple)
    target_cost = torch.zeros(inp.size(0))
    # if args.ggs_metric == 'lm' or args.ggs_metric == 'all':
    #     # target_trim = target[:, args.context_length:]
    #     target_cost = ggs_utils.task_distance(None, None, inp, score_model, 
    #                     "lm", tokenizer.eos_token_id, average=False)

    distances = []
    for i in range(inp.size(0)):
        decoded[f'context_{i}'] = (tokenizer.decode(batch[i, :args.context_length].tolist()), -1)
        decoded[f'target_{i}'] = (tokenizer.decode(batch[i].tolist()), target_cost[i])

    with timer_context('total_scoring_time') as ctxt_timer:
        start_time = timer()
        is_original, distance_curr, bpes_curr, output_curr, \
            decoded_samples = scoring_function(model, tokenizer, batch, score_model, 
                                        max_length, device, args, prefix='original')
        
        distance_curr_mean = distance_curr.mean(0).item()
        if args.efficient and not use_learned_scoring_func:
            idx = f'mgs_{args.log_step}'
            buffer.append(idx, InstanceType.NON_PERTURBED, batch_id, batch,
                            model, output_curr, distance_curr)

        if args.efficient and args.log_scoring_function and \
                            args.log_step % args.print_every == 1:
            _, distance_curr_score, _, _, _ = target_scoring_func(model, tokenizer, 
                    batch, score_model, max_length, device, args, prefix='original')
            distance_comp.append(('original', distance_curr, distance_curr_score))
            distance_curr_score_mean = distance_curr_score.mean(0).item()
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
            perturbed_models, log_rhos, noise_magnitudes, _, perturb_types = \
                ggs_utils.heuristic_perturb(
                    model, model_with_grad, args.ggs_num_samples, args.ggs_noise,
                    tokenizer, batch, score_model, device, args,
                    distance_curr_score,
                    noise_scale=args.noise_scale,
                    zero_dist_only=args.zero_dist_only,
                    mle_dist_only=args.mle_dist_only,
                    include_mle_gradient=args.include_mle_gradient
                )
        else:
            perturbed_models, log_rhos, noise_magnitudes, rng_states, perturb_types = \
                ggs_utils.perturb(model, model_with_grad, 
                    args.ggs_num_samples, args.ggs_noise,
                    noise_scale=args.noise_scale,
                    zero_dist_only=args.zero_dist_only,
                    mle_dist_only=args.mle_dist_only,
                    include_mle_gradient=args.include_mle_gradient)
        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    with timer_context('perturb_scoring_time') as perturb_scoring_timer, \
        timer_context('total_scoring_time') as total_scoring_timer:
        # -- Decode with perturbed models and compute task metric
        start_time = timer()
        distance_list = []
        distance_score_list = []
        for i, (p_model, rng_state, perturb_type) in \
                enumerate(zip(perturbed_models, rng_states, perturb_types)):

            is_original, perturb_distance, bpes, outputs, \
                decoded_samples = scoring_function(p_model, tokenizer, batch, score_model, 
                                            max_length, device, args, prefix=f'perturb_{i}')
            perturb_distance_mean = perturb_distance.mean(0).item()
            if args.efficient and not use_learned_scoring_func:
                # idx = f'mgs_perturb_{args.log_step}_{i}'
                idx = f'mgs_{args.log_step}'
                buffer.append(idx, InstanceType.PERTURBED, batch_id, batch, 
                                model, outputs, perturb_distance, rng_state=rng_state,
                                perturb_type=perturb_type)

            if args.efficient and args.log_scoring_function and \
                args.log_step % args.print_every == 1:
                _, perturb_distance_score, _, _,  _ = target_scoring_func(p_model, tokenizer, 
                    batch, score_model, max_length, device, args, prefix='perturb_{i}')
                perturb_distance_score_mean = perturb_distance_score.mean(0).item()
                distance_comp.append(('perturb_{i}', perturb_distance, perturb_distance_score))
                distance_score_list.append(perturb_distance_score_mean)

            distance_list.append(perturb_distance_mean)
            decoded.update(decoded_samples)
        end_time = timer()
        perturb_scoring_timer.timeit(start_time, end_time)
        total_scoring_timer.timeit(start_time, end_time)

    if args.efficient and args.log_scoring_function and \
            args.log_step % args.print_every == 1:
        log_weights = ggs_utils.compute_weight(distance_curr_mean, distance_list, log_rhos, args.ggs_beta)
        log_weights_score = ggs_utils.compute_weight(distance_curr_score_mean, distance_score_list, log_rhos, args.ggs_beta)
        kl_div = F.kl_div(log_weights, log_weights_score, log_target=True)
        logging.info(f"KL Distance: {kl_div:.3f}")


    # -- Compute weights
    # Kushal: please revise score_network(distance_curr - distances), where the scorer's output is embedding.
    with timer_context('weight_computation_time') as ctxt_timer:
        start_time = timer()
        log_weights = ggs_utils.compute_weight(distance_curr_mean, distance_list, log_rhos, args.ggs_beta)

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
        dagger_ggs_utils.MODEL_ID = ggs_utils.update(model, update_directions, 
                                    optimizer, args.max_grad_norm)

        end_time = timer()
        ctxt_timer.timeit(start_time, end_time)

    if is_original:
        with timer_context('metrics_update_time') as ctxt_timer:
            start_time = timer()
            # -- Record statistics
            metrics.step(
                mle_loss.item(), distance_curr_mean, bpes_curr, target, args.context_length,
                tokenizer.pad_token_id, tokenizer.eos_token_id,
                model_with_grad, update_directions, log_rhos, log_weights,
                noise_magnitudes, distance_list
            )
            end_time = timer()
            ctxt_timer.timeit(start_time, end_time)
    return is_original, decoded


def get_learning_function(score_network, step, args, total_num_batches):
    scoring_function = original_mgs_scoring_function
    target_scoring_func = partial(dagger_ggs_utils.dagger_mgs_scoring_function, score_network)

    use_learned_scoring_function = args.use_learned_scoring_function
    # For first 25% of the batches, always use original mgs function.
    if step < total_num_batches // 4:
        use_learned_scoring_function = False

    # For next 25% of the batches, use dagger_mgs for every 4th iteration.
    elif step < total_num_batches // 2 and step % 4 != 0:
        use_learned_scoring_function = False

    # For last 50% of the batches, use dagger mgs for 3 of the 4 iteration.
    elif step > total_num_batches // 2  and step % 4 == 0:
        use_learned_scoring_function = False

    if use_learned_scoring_function:
        scoring_function = partial(dagger_ggs_utils.dagger_mgs_scoring_function, score_network)
        target_scoring_func = original_mgs_scoring_function

    return scoring_function, target_scoring_func, use_learned_scoring_function

buffer = None

class MGSTrainingModule(LightningModule):
    def __init__(self, args, tokenizer, model, score_network, dataset_tensor_dict):
      super().__init__()
      self.args = args
      self.tokenizer = tokenizer
      self.model = model
      self.score_model = deepcopy(model)
      self.score_model.requires_grad_(False)
      self.buffer = buffer
      self.score_network = score_network
      self.dataset_tensor_dict = dataset_tensor_dict
      self.score_network_training_iter = 0
      self.automatic_optimization=False
      self.metrics = GuidedMetrics()

    def training_step(self, batch, batch_idx):
        with timer_context('total_step_time') as total_ctxt_timer:
            train_step_start = timer()
            total_num_batches = len(self.dataset_tensor_dict['train'])
            self.args.use_learned_scoring_function = False
            scoring_function, target_scoring_func, use_learned_scoring_func = \
                get_learning_function(self.score_network, self.global_step, self.args, 
                                                                total_num_batches)
            if self.args.efficient and \
                (self.global_step + 1) % self.args.retrain_score_network_every == 0:

                self.score_network_training_iter += 1
                score_network_trainer.train_score_network_lightning(self.buffer , 
                    self.score_network, self.tokenizer,self.args, self.score_network_training_iter, 
                       epochs=self.args.retrain_score_network_epochs, loggers=self.logger)

            batch_id = batch[0][0]
            batch = batch[1].squeeze(0)

            if batch.size(1) < self.args.context_length:
                return None

            with timer_context('total_mgs_time') as ctxt_timer:
                mgs_start_time = timer()
                using_orig_scoring_func, decoded = MGS(batch_id=batch_id, 
                        batch=batch, model=self.model, score_model=self.score_model, 
                        tokenizer=self.tokenizer, args=self.args, device=self.device,
                        metrics=self.metrics, optimizer=self.optimizers(),
                        scoring_function=scoring_function,
                        target_scoring_func=target_scoring_func,
                        use_learned_scoring_func=use_learned_scoring_func,
                        buffer=self.buffer)

                mgs_end_time = timer()
                ctxt_timer.timeit(mgs_start_time, mgs_end_time)

            train_step_end = timer()
            total_ctxt_timer.timeit(train_step_start, train_step_end)

        if using_orig_scoring_func and \
            self.global_step % self.args.print_every == 0:
            
            metrics_ = self.metrics.normalize('train')
            self.metrics.reset()
            logging.info("Epoch %d   \t Step %d   \tmle: %.3f\tdist: %.3f\tnon_term: %.3E\tmle_weight: %.3E" % (
                self.current_epoch,
                self.global_step,
                metrics_['train/mle_loss'],
                metrics_['train/distance'],
                metrics_['train/non_term'],
                metrics_['train/model/mle_weight']
            ))
            self.log_dict(metrics_, batch_size=1, on_epoch=True, on_step=True)

            if not self.args.use_learned_scoring_function \
                                and self.args.print_decodings:
                pidx = random.choice(range(batch.size(0)))
                print(f"Step: {self.global_step}")
                print(f"context(x_{pidx}):: {decoded['context_%d' % pidx][0]}")
                dec_and_dist = decoded['target_%d' % pidx]
                print(f"Target(x_{pidx}):: Cost: {dec_and_dist[1]:.2f} {dec_and_dist[0]}")
                dec_and_dist = decoded['original_%d' % pidx]
                print(f"theta(x_{pidx}):: Cost: {dec_and_dist[1]:.2f} {dec_and_dist[0]}")
                for j in range(self.args.ggs_num_samples):
                    dec_and_dist = decoded['perturb_%d_%d' % (j, pidx)]
                    print(f"theta+Delta_{j}(x_{pidx}):: Cost: {dec_and_dist[1]:.2f} {dec_and_dist[0]}")
                print('\n')


    def configure_optimizers(self):
        total_num_batches = len(self.dataset_tensor_dict['train'])
        optimizer, scheduler = utils.get_optimizer(self.model, total_num_batches, self.args)
        return [optimizer], [StepLR(optimizer, step_size=5, gamma=0.5, verbose=True)]

    def validation_epoch_end(self, outputs):
        val_loss, val_metrics, decodings = train_utils.valid_iteration(
            self.dataset_tensor_dict['valid'], self.model, self.score_model, 
            train_utils.get_mle_loss, self.tokenizer, self.device,
            context_length=self.args.eval_context_length,
            num_decodings=250,
            args=self.args)

        if self.args.print_decodings:
            logging.info(f"Validation Decodings at Step: {self.global_step}")
            prefixes = decodings['text_prefix']
            sentences = decodings['text_decoding_including_prefix']

            to_print_idxs = random.sample(range(len(sentences)), 10)
            for i in to_print_idxs:
                print(f"{'Prefix':10}: {prefixes[i]}")
                print(f"{'Sequence':10}: {sentences[i]}")                    
                print('\n')
        self.log("val_loss", val_loss, batch_size=1, prog_bar=True, on_epoch=True)
        self.log_dict(val_metrics, batch_size=1, on_epoch=True)

    def train_dataloader(self):
        train_sampler = RandomSampler(self.dataset_tensor_dict['train'])
        train_dataloader = DataLoader(
            self.dataset_tensor_dict['train'],
            sampler=train_sampler,
            batch_size=1)
        return train_dataloader

    def val_dataloader(self):
        valid_sampler = RandomSampler(self.dataset_tensor_dict['valid'])
        valid_dataloader = DataLoader(
            self.dataset_tensor_dict['valid'],
            sampler=valid_sampler,
            batch_size=1)
        return valid_dataloader


def train(model, tokenizer, dataset_tensor_dict, args, device):
    tensorboard_logger = TensorBoardLogger(
                            save_dir=args.save_base_dir)
    loggers = [tensorboard_logger]
    if args.wandb:
        wandb_logger = WandbLogger(
                    project=args.wandb_project_name,
                    save_dir=args.save_base_dir,
                    name=args.wandb_run_name,
                    entity='dagger_mgs',
                    tags=args.wandb_tags,
                    group="multigpu" if args.multigpu else None,
                    config=args)
        loggers.append(wandb_logger)

    # global ggs_utils.MODEL_ID
    dagger_ggs_utils.MODEL_ID = ggs_utils.get_model_id(model)
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    
    score_network_training_iter = 0
    score_network = None

    config = GPT2Config()
    score_model = deepcopy(model)
    total_num_batches = len(dataset_tensor_dict['train'])
    if args.efficient:
        # Initialize buffer and pretrain score network.

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

        # If using saved score network, use it, else accumulate training data, 
        # and train network on the accumulated data.

        score_network = score_network_utils.build_score_network(
                                        input_size=config.hidden_size, 
                                        args=args)
        score_network = score_network.to(device=device)
        
        if args.initialize_score_network:
            print('=' * 100)
            initial_training_epochs = args.retrain_score_network_epochs
            if not args.use_saved_score_network:
                initial_training_epochs = args.score_network_epochs

                if not args.use_saved_aggregated_data:
                    logging.info("Started Initial Data Accumulation.")

                    aggregated_data_size = min(args.aggregated_data_size, len(dataset_tensor_dict['train']))
                    aggregated_idxs = random.sample(range(len(dataset_tensor_dict['train'])), aggregated_data_size)
                    aggregated_dataset = []
                    if args.multigpu:
                        for step, idx in enumerate(aggregated_idxs):
                            aggregated_dataset.append(dataset_tensor_dict['train'][idx])

                        accumulator.start_scorer_training_data_accumulation(buffer, 
                                aggregated_dataset,  model, score_model, tokenizer, args)
                    else:
                        for step, idx in enumerate(aggregated_idxs):
                            aggregated_dataset.append(dataset_tensor_dict['train'][idx])
                            batch_id, batch = dataset_tensor_dict['train'][idx]
                            data = accumulator.accumulate_scorer_training_data(step, 
                                        batch_id, batch, model, score_model, tokenizer, args, device)
                            
                            buffer.append(*data['non_pert'])
                            for pert_data in data['pert']:
                                buffer.append(*pert_data)

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
                if True or args.multigpu:
                    score_network_trainer.train_score_network_lightning(buffer , score_network, tokenizer,
                                    args, score_network_training_iter, epochs=initial_training_epochs,
                                    loggers=loggers)
                else:
                    train_scorer_dataloader, valid_scorer_dataloader = \
                            buffer.get_dataloaders(args, device)
                    score_network_trainer.train_score_network(device, score_network, tokenizer, 
                                            train_scorer_dataloader, valid_scorer_dataloader, args, 
                                            score_network_training_iter, epochs=initial_training_epochs)
                print('=' * 100)
    
                if args.only_train_score_network:
                    return

    model.train()

    torch.multiprocessing.set_sharing_strategy('file_system')
    module = MGSTrainingModule(args, tokenizer, model,
                                score_network, dataset_tensor_dict)

    strategy = None
    if args.multigpu: 
        strategy = DDPSpawnPlugin(find_unused_parameters=False)
    trainer = Trainer(
                gpus=-1 if args.multigpu else 1,
                default_root_dir=args.save_base_dir,
                strategy=strategy,
                max_epochs=args.num_train_epochs,
                num_sanity_val_steps=0, 
                # callbacks=RichProgressBar(),
                logger=loggers)
        
    trainer.fit(model=module)

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
    for epoch_number in range(args.num_train_epochs):
        metrics = GuidedMetrics()

        for step, (batch_id, batch) in enumerate(train_dataloader):
            with timer_context('total_step_time') as total_ctxt_timer:
                train_step_start = timer()

                scoring_function, target_scoring_func, use_learned_scoring_func = \
                    get_learning_function(score_network, step, args, total_num_batches)

                if args.efficient:
                    # if dagger_ggs_utils.shall_accumulate_scorer_training_data(step, total_num_batches, args):
                    #     dagger_ggs_utils.accumulate_scorer_training_data(step, batch_id, 
                    #                                     batch, buffer, model, score_model, 
                    #                                     tokenizer, args, device)

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
                    using_orig_scoring_func, decoded = MGS(batch_id=batch_id, 
                            batch=batch, model=model, score_model=score_model, 
                            tokenizer=tokenizer, args=args, device=device,
                            metrics=metrics, optimizer=optimizer,
                            scoring_function=scoring_function,
                            target_scoring_func=target_scoring_func,
                            use_learned_scoring_func=use_learned_scoring_func,
                            buffer=buffer)

                    mgs_end_time = timer()
                    ctxt_timer.timeit(mgs_start_time, mgs_end_time)

                train_step_end = timer()
                total_ctxt_timer.timeit(train_step_start, train_step_end)

            if using_orig_scoring_func and step % args.print_every == 0:
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
        choices=['edit', 'lm', 'all'],
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
        "--multigpu", action='store_true',
    )

    parser.add_argument('--include-mle-gradient',
                                action='store_true')
    parser = dagger_ggs_utils.add_args(parser)
    parser = score_network_trainer.add_args(parser)
    parser = accumulator.add_args(parser)

    return parser
