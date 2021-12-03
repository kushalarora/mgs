import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from others.train_utils import load_model, get_optimizer, get_mle_loss, decode_and_distance
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.eval_utils import evaluate, decode
from others.data_utils import load_dataset
from others.args_utils import add_args


class Baseline(object):
    def __init__(self, alpha=0.95):

        self.alpha = alpha
        self.value = 0.

    def update(self, reward):
        self.value = (1.0 - self.alpha) * self.value + self.alpha * reward.mean().item()

    def __call__(self, *args, **kwargs):
        return self.value


def get_pg_args():
    parser = argparse.ArgumentParser(description='Policy gradient.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--stochastic_decoding', type=str, default='temp-1.0'
    )
    parser.add_argument(
        '--metric', type=str, choices=['edit', 'lm'], default='lm'
    )
    parser.add_argument(
        '--pg_mle_mix', type=float, default=0.1
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


def get_pg_loss(model, batch, candidates, distances, baseline, eos_idx, args):

    # Current model score
    log_probs = torch.log_softmax(model(candidates)[0], -1)
    log_probs = log_probs[:, :-1, :].gather(2, candidates[:, 1:].unsqueeze(-1)).squeeze(-1)
    seq_masks = (candidates[:, 1:].eq(eos_idx).cumsum(1).cumsum(1) <= 1).float()

    # Reshape to batch_size x num_candidates x -1
    batch_size = batch.size(0)
    log_probs = log_probs.view(batch_size, args.num_samples, -1)
    seq_masks = seq_masks.view(batch_size, args.num_samples, -1)

    # Sequence-level evaluation score
    distances = distances.view(batch_size, args.num_samples)

    # For large-magnitude distances, e.g., language modeling score
    if args.normalized_distance:
        distances = distances / distances.max()

    reward = -distances
    baseline.update(reward)
    b = baseline(model, batch)

    scores = (log_probs * seq_masks).sum(2) / seq_masks.sum(2)
    loss = -((reward - b) * scores).mean(1).sum()
    distance = distances.mean(1).sum()

    metrics = {
        'loss': loss.item(),
        'distance': distance.item(),
        'num_docs': batch_size
    }

    return loss, metrics


def main():
    device = torch.device('cuda')

    parser = get_pg_args()
    args = add_args(parser)
    logger, exp_path = setup('PG_{}_{}_{}'.format(args.metric, args.normalized_distance, args.pg_mle_mix), args)

    logger.info('Build model.')
    tokenizer, pad_idx, eos_idx, model, score_mle_model = load_model(args, mode='train')
    logger.info('Initialized modules with %.2fM parameters.' % count_parameters(model))

    logger.info('Load data.')
    dataset = load_dataset(pad_idx, args)
    train_dataloader = DataLoader(dataset=dataset['train'], sampler=RandomSampler(dataset['train']))
    valid_dataloader = DataLoader(dataset=dataset['valid'], sampler=SequentialSampler(dataset['valid']))

    baseline = Baseline()

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

            if torch.rand(1).item() < args.pg_mle_mix:
                loss, _ = get_mle_loss(model, batch, pad_idx)
            else:
                # Decode candidates
                _, _, full_decodings, distances = decode_and_distance(
                    model, score_mle_model, tokenizer, batch, args, deterministic=False, average=False
                )

                # Sequence-level loss
                loss, seq_metrics = get_pg_loss(
                    model, batch, full_decodings, distances, baseline, eos_idx, args
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
                logger.info('Epoch: %d, Batch: [%d/%d], Loss: %.3f, %s: %.3f'
                            % (epoch, idx, args.num_batches, cur_loss, args.metric, cur_seq_metric))

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
