import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from copy import deepcopy
from others.train_utils import load_model, get_optimizer, get_mle_loss, decode_and_distance
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.mgs_utils import get_mle_gradient, get_weighted_directions
from others.eval_utils import evaluate, decode
from others.data_utils import load_dataset
from others.args_utils import add_args


def get_es_args():
    parser = argparse.ArgumentParser(description='Evolution strategy.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--metric', type=str, choices=['edit', 'lm'], default='lm'
    )
    parser.add_argument(
        '--num_directions', type=int, default=4
    )
    parser.add_argument(
        '--valid_every', type=int, default=1000
    )
    return parser


def perturb(model, model_with_grad, score_mle_model, tokenizer, batch, args):

    perturbed_models = []
    distances = []

    for idx in range(args.num_directions):

        model_ = deepcopy(model)

        for param, (name, param_with_grad) in zip(model_.parameters(), model_with_grad.named_parameters()):
            gradient = -param_with_grad.grad.data
            noise_ = torch.randn_like(param.data) * (gradient.abs().sum() / gradient.numel())
            param.data = param.data + noise_

        # Decode with the #num_directions perturbed modules and compute the sequence-level evaluation score
        _, _, _, per_distances = decode_and_distance(
            model_, score_mle_model, tokenizer, batch, args, deterministic=True, average=True
        )
        if args.metric != 'lm':
            per_distances = torch.as_tensor(per_distances).to(batch.device)

        perturbed_models.append(model_)
        distances.append(per_distances)

    distances = torch.stack(distances)

    return perturbed_models, distances


def main():
    device = torch.device('cuda')

    parser = get_es_args()
    args = add_args(parser)
    logger, exp_path = setup('ES_{}'.format(args.metric), args)

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
            model_with_gradient, batch_metrics = get_mle_gradient(current_model, batch, pad_idx)

            perturbed_models, distances = perturb(model, model_with_gradient, score_mle_model, tokenizer, batch, args)
            weights = torch.clamp(cur_distances.mean() - distances, max=1e16)

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

            if total_steps % args.valid_every == 0:
                val_loss, val_ppl, val_metrics = evaluate(model, score_mle_model, tokenizer, valid_dataloader, args)
                val_seq_metric = val_metrics['valid/%s' % args.metric]
                logger.info('Loss: %.3f, PPL: %.3f, %s: %.3f' % (val_loss, val_ppl, args.metric, val_seq_metric))

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
