import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from copy import deepcopy
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.train_utils import load_model, get_optimizer, decode_and_distance
from others.mgs_utils import get_mle_gradient, perturb
from others.eval_utils import evaluate, decode
from others.proxy_utils import ProxyDataset
from others.data_utils import load_dataset
from others.args_utils import add_args


def get_train_args():
    parser = argparse.ArgumentParser(description='Efficient parameter search.')
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
        '--wandb_tags', nargs='+', type=str
    )
    parser.add_argument(
        '--proxy_batch_size', type=int, default=64
    )
    parser.add_argument(
        '--proxy_max_epochs', type=int, default=10
    )
    parser.add_argument(
        '--proxy_lr', type=float, default=2.5e-4
    )
    parser.add_argument(
        '--proxy_adam_epsilon', type=float, default=1e-8
    )
    parser.add_argument(
        '--proxy_weight_decay', type=float, default=1e-6
    )
    parser.add_argument(
        '--proxy_warmup_steps', type=int, default=0
    )
    parser.add_argument(
        '--proxy_network_type', type=str,
        choices=['simple_mlp', 'simple_mlp_w_relu', 'simple_mlp_w_residual', 'simple_mlp_w_layer_norm',
                 'simple_mlp_w_predictions_v2', 'simple_mlp_w_targets',  'simple_mlp_w_targets_v2',
                 'simple_mlp_w_predictions_targets_v2', 'simple_mlp_w_targets_v3', 'simple_mlp_w_targets_v4',
                 'simple_mlp_complete_context', 'simple_mlp_complete_context_v2'],
        default='simple_mlp_w_targets_v2',
    )
    parser.add_argument(
        '--proxy_network_hidden_size', type=int, default=2048,
    )
    parser.add_argument(
        '--proxy_network_num_layers', type=int, default=3,
    )
    parser.add_argument(
        '--proxy_network_dropout', type=float, default=0.1,
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

    parser = get_train_args()
    args = add_args(parser)
    logger, wandb_run, exp_path = setup('eMGS_{}_{}_{}_{}_{}'.format(args.metric,
                                                                     args.num_directions,
                                                                     args.zero_dist,
                                                                     args.mle_dist,
                                                                     args.beta), args)

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

    proxy_dataset = ProxyDataset()

    with wandb_run:

        # Train proxy

        logger.info('Start training proxy network.')

        # Train generative model and proxy

        logger.info('Start training GPT-2 and proxy network.')

        for epoch in range(1, args.max_epochs + 1):

            train_loss = 0.
            num_tokens = 0
            seq_metric = 0.
            num_docs = 0

            for step, batch in enumerate(train_dataloader):

                model.train()

                batch = batch.squeeze(0)
                batch = batch.to(device)
                assert batch.size(1) >= args.context_length + 1

                # Decode with the current MLE model and compute the sequence-level evaluation score
                _, _, _, cur_distances = decode_and_distance(
                    model, score_mle_model, tokenizer, batch, args, deterministic=True, average=False
                )

                if total_steps % args.valid_every == 0:
                    val_loss, val_ppl, val_metrics = evaluate(model, score_mle_model, tokenizer, valid_dataloader, args)
                    val_seq_metric = val_metrics['valid/%s' % args.metric]
                    logger.info(
                        '[Valid] Loss: %.3f, PPL: %.3f, %s: %.3f' % (val_loss, val_ppl, args.metric, val_seq_metric)
                    )
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
