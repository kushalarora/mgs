import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from others.train_utils import load_model, get_optimizer, generate_batch, trim_batch
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.metrics import GenerationMetrics
from others.data_utils import load_dataset
from others.args_utils import add_args


def get_st_args():
    parser = argparse.ArgumentParser(description='Self-terminating GPT-2.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--epsilon_upper_bound', type=float, default=0.001
    )
    parser.add_argument(
        '--valid_every', type=int, default=1000
    )
    return parser


def get_st_mle_loss(model, batch, epsilon_upper_bound, pad_idx, eos_idx):
    source = batch[:, :-1]
    source_ = source.clone()
    source_[source == pad_idx] = 0

    target = batch[:, 1:]
    target_ = target.clone()
    target_[target == pad_idx] = 0

    source_mask = source.ne(pad_idx).float()
    target_mask = target.ne(pad_idx).float()

    output = model(source_, attention_mask=source_mask)
    logits = output.logits

    # Compute the log-softmax based on the self-terminating algorithm
    log_probs = model.st_softmax(logits, eos_idx, epsilon_upper_bound)

    loss = F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        target_.reshape(-1),
        reduction='none'
    )
    loss = loss * target_mask.view(loss.size())
    loss = loss.sum()
    num_tokens = target_mask.sum()

    metrics = {
        'loss': loss.item(),
        'num_tokens': num_tokens.item()
    }

    return loss, metrics


def evaluate(model, data_source, pad_idx, eos_idx, args):
    model.eval()

    total_loss = 0.
    num_tokens = 0

    with torch.no_grad():

        for _, batch in enumerate(data_source):

            batch = batch.squeeze(0)
            batch = batch.to(model.device)

            if (batch[:, :args.eval_context_length + 1] == pad_idx).sum() > 0:
                continue

            _, batch_metrics = get_st_mle_loss(model, batch, args.epsilon_upper_bound, pad_idx, eos_idx)
            total_loss += batch_metrics['loss']
            num_tokens += batch_metrics['num_tokens']

        report_loss = total_loss / num_tokens
        report_ppl = np.exp(report_loss)

    return report_loss, report_ppl


def decode(model, score_model, tokenizer, data_source, args):
    model.eval()

    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id

    total_loss = 0.
    num_tokens = 0

    gen_metrics = GenerationMetrics()

    bpe_prefixes = []
    txt_prefixes = []
    bpe_target_continuations = []
    bpe_output_continuations = []
    bpe_output_including_prefixes = []
    txt_output_including_prefixes = []

    with torch.no_grad():

        for _, batch in enumerate(data_source):

            batch = batch.squeeze(0)
            batch = batch.to(model.device)

            if (batch[:, :args.eval_context_length + 1] == pad_idx).sum() > 0:
                continue

            _, batch_metrics = get_st_mle_loss(model, batch, args.epsilon_upper_bound, pad_idx, eos_idx)
            total_loss += batch_metrics['loss']
            num_tokens += batch_metrics['num_tokens']

            bpe_prefix, txt_prefix, bpe_decoding, txt_decoding, full_decodings = generate_batch(
                model, tokenizer, batch, args
            )
            bpe_decodings = [bpe[args.eval_context_length:] for bpe in bpe_decoding]
            targets_trim, outputs_trim = trim_batch(batch, bpe_decodings, args.eval_context_length, eos_idx)
            gen_metrics.step(targets_trim, outputs_trim, full_decodings, score_model, eos_idx)

            bpe_prefixes.extend(bpe_prefix)
            txt_prefixes.extend(txt_prefix)
            bpe_target_continuations.extend(targets_trim)
            bpe_output_continuations.extend(outputs_trim)
            bpe_output_including_prefixes.extend(bpe_decoding)
            txt_output_including_prefixes.extend(txt_decoding)

        report_loss = total_loss / num_tokens
        report_ppl = np.exp(report_loss)

        metrics = {}
        gen_metrics = gen_metrics.normalize('test')
        for k, v in gen_metrics.items():
            metrics[k] = v

        decodings = {
            'bpe_prefixes': bpe_prefixes,
            'txt_prefixes': txt_prefixes,
            'bpe_target_continuations': bpe_target_continuations,
            'bpe_output_continuations': bpe_output_continuations,
            'bpe_output_including_prefixes': bpe_output_including_prefixes,
            'txt_output_including_prefixes': txt_output_including_prefixes
        }

    return report_loss, report_ppl, metrics, decodings


def main():
    device = torch.device('cuda')

    parser = get_st_args()
    args = add_args(parser)
    logger, exp_path = setup('ST', args)

    logger.info('Build model.')
    tokenizer, pad_idx, eos_idx, model = load_model(args, mode='train', name='ST')
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

            loss, batch_metrics = get_st_mle_loss(model, batch, args.epsilon_upper_bound, pad_idx, eos_idx)

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
                val_loss, val_ppl = evaluate(model, valid_dataloader, pad_idx, eos_idx, args)
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

    model, score_mle_model = load_model(args, mode='test', name='ST')
    test_dataloader = DataLoader(dataset=dataset['test'], sampler=SequentialSampler(dataset['test']))
    test_loss, test_ppl, test_metrics, decodings = decode(model, score_mle_model, tokenizer, test_dataloader, args)
    logger.info('[Test] Loss: %.3f, PPL: %.3f' % (test_loss, test_ppl))

    logger.info('Test results:')
    for k, v in test_metrics.items():
        logger.info('\t%s\t%.3f' % (k, v))

    save_results(test_metrics, decodings, distances=None, save_path=exp_path)


if __name__ == '__main__':
    main()
