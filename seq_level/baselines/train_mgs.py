import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from copy import deepcopy
from others.mgs_utils import (get_mle_gradient, perturb, heuristic_perturb,
                              compute_weights, get_weighted_directions,
                              weight_metrics)
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.train_utils import load_model, get_optimizer, decode_and_distance
from others.eval_utils import evaluate, decode
from others.data_utils import load_dataset
from others.args_utils import add_args


def get_mgs_args():
    parser = argparse.ArgumentParser(description='MLE-guided parameter search.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--wandb_project_name', type=str, default='efficient_mgs'
    )
    parser.add_argument(
        '--wandb_run_name', type=str, default=None
    )
    parser.add_argument(
        '--wandb_tags', type=str, nargs='+'
    )
    parser.add_argument(
        '--beta', type=float, default=1.0
    )
    parser.add_argument(
        '--heuristic', type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        '--metric', type=str, choices=['edit', 'lm'], default='lm'
    )
    parser.add_argument(
        '--num_directions', type=int, default=4
    )
    parser.add_argument(
        '--noise', type=float, default=1.0
    )
    parser.add_argument(
        '--noise_scale', type=str, choices=['uniform', 'constant'], default='uniform'
    )
    parser.add_argument(
        '--zero_dist', type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        '--mle_dist', type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        '--valid_every', type=int, default=1000
    )
    return parser


def main():
    device = torch.device('cuda')

    parser = get_mgs_args()
    args = add_args(parser)
    logger, wandb_run, exp_path = setup('MGS_{}_{}_{}_{}'.format(args.metric,
                                                                 args.num_directions,
                                                                 args.zero_dist,
                                                                 args.mle_dist), args)

    logger.info('Build model.')
    tokenizer, pad_idx, eos_idx, model, score_mle_model = load_model(args, mode='train')
    logger.info('Initialized modules with %.2fM parameters.' % count_parameters(model))

    logger.info('Load data.')
    dataset = load_dataset(pad_idx, args)
    train_dataloader = DataLoader(dataset=dataset['train'], sampler=RandomSampler(dataset['train']))
    valid_dataloader = DataLoader(dataset=dataset['valid'], sampler=SequentialSampler(dataset['valid']))

    args.num_batches = len(train_dataloader)
    total_steps = 1
    best_val_seq_metric = 1e4
    patience = args.patience

    logger.info('Build optimizer.')
    optimizer, _ = get_optimizer(model, args)

    with wandb_run:

        if args.heuristic:
            logger.info('Start heuristic training.')
        else:
            logger.info('Start training.')

        for epoch in range(1, args.max_epochs + 1):

            train_loss = 0.
            num_tokens = 0
            seq_metric = 0.
            num_docs = 0

            for idx, batch in enumerate(train_dataloader):

                model.train()

                batch = batch.squeeze(0)
                batch = batch.to(device)
                assert batch.size(1) >= args.context_length + 1

                # Decode with the current MLE model and compute the sequence-level evaluation score
                _, _, _, cur_distances = decode_and_distance(
                    model, score_mle_model, tokenizer, batch, args, deterministic=True, average=False
                )

                # Get the current MLE gradients
                current_model = deepcopy(model)
                model_with_gradient, batch_metrics = get_mle_gradient(current_model, batch, pad_idx, args.max_grad_norm)

                # Inject randomness into the current MLE gradients
                if not args.heuristic:
                    perturbed_models, log_q_dists, distances = perturb(
                        model, model_with_gradient, score_mle_model, tokenizer, batch, args
                    )
                else:
                    perturbed_models, log_q_dists, distances = heuristic_perturb(
                        model, model_with_gradient, score_mle_model, tokenizer, batch, cur_distances, args
                    )

                # Compute average weights
                weights = compute_weights(cur_distances.mean(), distances, log_q_dists, args.beta)
                # weights_metrics = weight_metrics(weights)
                # wandb_run.log({'mle': weights_metrics['mle_weight'], 'zero': weights_metrics['zero_weight']})

                # Compute the weighted average of #num_directions directions
                updated_directions = get_weighted_directions(model, perturbed_models, weights)

                # Perform the parameter update
                optimizer.zero_grad()

                for name, param in model.named_parameters():
                    param.grad = -updated_directions[name]

                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                train_loss += batch_metrics['loss']
                num_tokens += batch_metrics['num_tokens']
                seq_metric += cur_distances.sum().item()
                num_docs += batch_metrics['num_docs']

                if idx % args.print_every == 0 and idx > 0:
                    cur_loss = train_loss / num_tokens
                    cur_seq_metric = seq_metric / num_docs
                    logger.info('Epoch: %d, Batch: [%d/%d], Loss: %.3f, PPL: %.3f, %s: %.3f' % (
                        epoch, idx, args.num_batches, cur_loss, np.exp(cur_loss), args.metric, cur_seq_metric))
                    wandb_run.log({'train/Loss': cur_loss, 'train/%s' % args.metric: cur_seq_metric})

                if total_steps % args.valid_every == 0:
                    val_loss, val_ppl, val_metrics = evaluate(model, score_mle_model, tokenizer, valid_dataloader, args)
                    val_seq_metric = val_metrics['valid/%s' % args.metric]
                    logger.info('Loss: %.3f, PPL: %.3f, %s: %.3f' % (val_loss, val_ppl, args.metric, val_seq_metric))
                    wandb_run.log({'valid/Loss': val_loss, 'valid/%s' % args.metric: val_seq_metric})

                    if val_seq_metric < best_val_seq_metric:
                        logger.info('Update best sequence-level score [%s]: [%.3f].' % (args.metric, val_seq_metric))
                        best_val_seq_metric = val_seq_metric
                        patience = args.patience
                        if not args.no_checkpoint:
                            logger.info('Save the model.')
                            save_model(model, exp_path)
                    else:
                        patience = patience - 1

                total_steps += 1

                if patience == 0:
                    break

                torch.cuda.empty_cache()

    logger.info('Start testing.')

    model, score_mle_model = load_model(args, mode='test')
    test_dataloader = DataLoader(dataset=dataset['test'], sampler=SequentialSampler(dataset['test']))
    test_loss, test_ppl, test_metrics, decodings = decode(model, score_mle_model, tokenizer, test_dataloader, args)
    test_seq_metric = test_metrics['test/%s' % args.metric]
    logger.info('[Test] Loss: %.3f, PPL: %.3f, %s: %.3f' % (test_loss, test_ppl, args.metric, test_seq_metric))

    logger.info('Test results:')
    for k, v in test_metrics.items():
        logger.info('\t%s\t%.3f' % (k, v))

    save_results(test_metrics, decodings, distances=None, save_path=exp_path)


if __name__ == '__main__':
    main()
