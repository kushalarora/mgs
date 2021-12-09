import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.train_utils import load_model, get_optimizer, get_mle_loss
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


def get_gold_args():
    parser = argparse.ArgumentParser(description='Text generation by learning from demonstrations.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--metric', type=str, choices=['lm'], default='lm'
    )
    parser.add_argument(
        '--valid_every', type=int, default=1000
    )
    return parser


def get_gold_loss(model, score_model, batch, baseline, pad_idx, eos_idx):

    batch_ = batch.clone()
    batch_[batch == pad_idx] = 0

    seq_masks = (batch_[:, 1:].eq(eos_idx).cumsum(1).cumsum(1) <= 1).float()

    log_probs_model = torch.log_softmax(model(batch_)[0], -1)
    log_probs_model = log_probs_model[:, :-1, :].gather(2, batch_[:, 1:].unsqueeze(-1)).squeeze(-1)

    score_model.eval()
    with torch.no_grad():
        log_probs_mle = torch.log_softmax(score_model(batch_)[0], -1)
        log_probs_mle = log_probs_mle[:, :-1, :].gather(2, batch_[:, 1:].unsqueeze(-1)).squeeze(-1)
        distances = -(log_probs_mle * seq_masks).sum(1)
    distances = distances / distances.max()

    reward = -distances
    baseline.update(reward)
    b = baseline(model, batch)

    scores = (log_probs_model * seq_masks).sum(1) / seq_masks.sum(1)
    loss = -((reward - b) * scores).sum()
    distance = distances.sum()

    metrics = {
        'loss': loss.item(),
        'distance': distance.item(),
        'num_docs': batch.size(0)
    }

    return loss, metrics


def main():
    device = torch.device('cuda')

    parser = get_gold_args()
    args = add_args(parser)
    logger, exp_path = setup('GOLD', args)

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

            # Off-policy GOLD: inputs are essentially ground-truth sequences
            loss, seq_metrics = get_gold_loss(
                model,
                score_mle_model,
                batch,
                baseline,
                pad_idx,
                eos_idx
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
