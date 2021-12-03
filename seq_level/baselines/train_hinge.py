import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from others.train_utils import load_model, get_optimizer, get_mle_loss, decode_and_distance
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.eval_utils import evaluate, decode
from others.data_utils import load_dataset
from others.args_utils import add_args


def get_hinge_args():
    parser = argparse.ArgumentParser(description='Hinge loss.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--stochastic_decoding', type=str, default='beam-5'
    )
    parser.add_argument(
        '--criterion', type=str, choices=['MaxMargin', 'MultiMargin', 'SoftMaxMultiMargin'], default='MaxMargin'
    )
    parser.add_argument(
        '--metric', type=str, choices=['edit', 'lm'], default='lm'
    )
    parser.add_argument(
        '--hg_mle_mix', type=float, default=0.4
    )
    parser.add_argument(
        '--num_samples', type=int, default=4
    )
    parser.add_argument(
        '--normalized_distance', type=int, choices=[0, 1], default=1
    )
    parser.add_argument(
        '--include_greedy', type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        '--include_target', type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        '--valid_every', type=int, default=1000
    )
    return parser


def get_hinge_loss(model, candidates, distances, batch_size, eos_idx, args):

    # Current model score
    log_probs = torch.log_softmax(model(candidates)[0], -1)
    log_probs = log_probs[:, :-1, :].gather(2, candidates[:, 1:].unsqueeze(-1)).squeeze(-1)
    seq_masks = (candidates[:, 1:].eq(eos_idx).cumsum(1).cumsum(1) <= 1).float()

    # Reshape to batch_size x num_candidates x -1
    log_probs = log_probs.view(batch_size, args.num_samples, -1)
    seq_masks = seq_masks.view(batch_size, args.num_samples, -1)

    # Sequence-level evaluation score
    distances = distances.view(batch_size, args.num_samples)

    # For large-magnitude distances, e.g. language modeling score = log p(y) = -distances
    if args.normalized_distance:
        distances = distances / distances.max()

    reward = -distances
    scores = (log_probs * seq_masks).sum(2) / seq_masks.sum(2)
    values = scores - reward

    max_reward_indices = torch.max(reward, 1)[1]
    max_scores_indices = torch.max(scores, 1)[1]

    if args.criterion == 'MaxMargin':
        values_with_high_target = values.clone()
        values_with_high_target.scatter_(1, max_reward_indices.view(-1, 1), 1e5)
        values_with_high_target.scatter_(1, max_scores_indices.view(-1, 1), 1e3)
        target_and_offender_index = values_with_high_target.sort(1, True)[1][:, 0:2]
        values = values.gather(1, target_and_offender_index)

        target_indices = torch.zeros(max_reward_indices.size(), dtype=torch.long, device=candidates.device)

        loss = F.multi_margin_loss(
            input=values,
            target=target_indices,
            margin=0,
            reduction='none'
        )

    elif args.criterion == 'MultiMargin':
        loss = F.multi_margin_loss(
            input=values,
            target=max_reward_indices,
            margin=0,
            reduction='none'
        )

    elif args.criterion == 'SoftMaxMultiMargin':
        loss = F.cross_entropy(
            input=values,
            target=max_reward_indices,
            reduction='none'
        )

    else:
        raise NotImplementedError('%s does not match any known criterion.' % args.criterion)

    loss = loss.sum()
    distance = distances.mean(1).sum()

    metrics = {
        'loss': loss.item(),
        'distance': distance.item(),
        'num_docs': batch_size,
    }

    return loss, metrics


def main():
    device = torch.device('cuda')

    parser = get_hinge_args()
    args = add_args(parser)

    logger, exp_path = setup('HG_{}_{}_{}_{}'.format(args.metric,
                                                     args.criterion,
                                                     args.normalized_distance,
                                                     args.hg_mle_mix), args)

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
    optimizer, scheduler = get_optimizer(model, args)

    logger.info('Start training.')

    for epoch in range(1, args.max_epochs + 1):

        train_loss = 0.
        seq_metric = 0.
        num_docs = 0

        for idx, batch in enumerate(train_dataloader):

            model.train()

            batch = batch.squeeze(0)
            batch = batch.to(device)
            assert batch.size(1) >= args.context_length + 1

            optimizer.zero_grad()

            if torch.rand(1).item() < args.hg_mle_mix:
                loss, _ = get_mle_loss(model, batch, pad_idx)
            else:
                # Decode candidates
                _, _, full_decodings, distances = decode_and_distance(
                    model, score_mle_model, tokenizer, batch, args, deterministic=False, average=False
                )

                # Sequence-level loss
                loss, seq_metrics = get_hinge_loss(
                    model, full_decodings, distances, batch.size(0), eos_idx, args
                )

                train_loss += seq_metrics['loss']
                seq_metric += seq_metrics['distance']
                num_docs += seq_metrics['num_docs']

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if idx % args.print_every == 0 and idx > 0:
                cur_loss = train_loss / num_docs
                cur_seq_metric = seq_metric / num_docs
                logger.info('Epoch: %d, Batch: [%d/%d], Loss: %.3f, %s: %.3f' % (
                    epoch, idx, args.num_batches, cur_loss, args.metric, cur_seq_metric))

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
