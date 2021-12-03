import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.train_utils import load_model, get_optimizer, get_mle_loss
from others.data_utils import load_dataset
from others.args_utils import add_args
from others.eval_utils import decode


def get_mle_args():
    parser = argparse.ArgumentParser(description='Fine-tune starting from GPT-2.')
    parser.add_argument(
        '--model_name', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2'
    )
    parser.add_argument(
        '--valid_every', type=int, default=5000
    )
    return parser


def evaluate(model, data_source, pad_idx, args):
    total_loss = 0.
    num_tokens = 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(data_source):
            batch = batch.squeeze(0)
            batch = batch.to(model.device)
            if (batch[:, :args.eval_context_length + 1] == pad_idx).sum() > 0:
                continue
            _, batch_metrics = get_mle_loss(model, batch, pad_idx)
            total_loss += batch_metrics['loss']
            num_tokens += batch_metrics['num_tokens']
        report_loss = total_loss / num_tokens
        report_ppl = np.exp(report_loss)
    return report_loss, report_ppl


def main():
    device = torch.device('cuda')

    parser = get_mle_args()
    args = add_args(parser)
    logger, exp_path = setup('MLE', args)

    logger.info('Build model.')
    tokenizer, pad_idx, eos_idx, model = load_model(args, mode='train')
    logger.info('Initialized modules with %.2fM parameters.' % count_parameters(model))

    logger.info('Load data.')
    dataset = load_dataset(pad_idx, args)
    train_dataloader = DataLoader(dataset=dataset['train'], sampler=RandomSampler(dataset['train']))
    valid_dataloader = DataLoader(dataset=dataset['valid'], sampler=SequentialSampler(dataset['valid']))

    args.num_batches = len(train_dataloader)
    total_steps = 1
    best_val_loss = 1e4
    patience = args.patience

    logger.info('Build optimizer.')
    optimizer, scheduler = get_optimizer(model, args)

    logger.info('Start training.')

    for epoch in range(1, args.max_epochs + 1):

        train_loss = 0.
        num_tokens = 0

        for idx, batch in enumerate(train_dataloader):

            model.train()

            batch = batch.squeeze(0)
            batch = batch.to(device)
            assert batch.size(1) >= args.context_length + 1

            optimizer.zero_grad()

            loss, batch_metrics = get_mle_loss(model, batch, pad_idx)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += batch_metrics['loss']
            num_tokens += batch_metrics['num_tokens']

            if idx % args.print_every == 0 and idx > 0:
                cur_loss = train_loss / num_tokens
                logger.info('Epoch: %d, Batch: [%d/%d], Loss: %.3f, PPL: %.3f'
                            % (epoch, idx, args.num_batches, cur_loss, np.exp(cur_loss)))

            if total_steps % args.valid_every == 0:
                val_loss, val_ppl = evaluate(model, valid_dataloader, pad_idx, args)
                logger.info('Loss: %.3f, PPL: %.3f' % (val_loss, val_ppl))

                if val_loss < best_val_loss:
                    logger.info('Update best loss: [%.3f].' % val_loss)
                    best_val_loss = val_loss
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
    logger.info('[Test] Loss: %.3f, PPL: %.3f' % (test_loss, test_ppl))

    logger.info('Test results:')
    for k, v in test_metrics.items():
        logger.info('\t%s\t%.3f' % (k, v))

    save_results(test_metrics, decodings, distances=None, save_path=exp_path)


if __name__ == '__main__':
    main()
