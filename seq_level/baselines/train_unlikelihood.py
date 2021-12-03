import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from others.train_utils import (load_model, get_optimizer, get_mle_loss,
                                wrap_context_batch, set_max_length, generate_batch, trim_batch)
from others.exp_utils import setup, count_parameters, save_model, save_results
from others.data_utils import load_dataset
from others.args_utils import add_args
from others.eval_utils import decode


def get_unlikelihood_args():
    parser = argparse.ArgumentParser(description='Unlikelihood training.')
    parser.add_argument(
        '--wandb', action='store_true'
    )
    parser.add_argument(
        '--top_k', type=int, default=1
    )
    parser.add_argument(
        '--top_p', type=float, default=0.
    )
    parser.add_argument(
        '--seq_ul_mix', type=float, default=0.5
    )
    parser.add_argument(
        '--seq_n_grams', type=int, default=4
    )
    parser.add_argument(
        '--valid_every', type=int, default=1000
    )
    return parser


def top_k_top_p_filtering(logits, top_k=0, top_p=0., filter_value=-float('Inf')):

    assert logits.size(0) == 1
    logits = logits.squeeze(0)
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def ngram_repeat_mask(sequences, n, eos_idx):
    mask = torch.zeros_like(sequences)
    for i, sequence in enumerate(sequences):
        x = sequence.tolist()
        x = x[:x.index(eos_idx) + 1] if eos_idx in x else x
        seen = set()
        for j in range(len(x) - n):
            ng = tuple(x[j:j + n])
            if ng in seen:
                mask[i, j:j + n] = 1
            seen.add(ng)
    return mask


def get_seq_loss(model, batch, pad_idx, eos_idx, args):

    context_batch = wrap_context_batch(batch, args)
    context_batch = context_batch.to(batch.device)

    if context_batch.size(0) > 0:

        model.eval()

        if args.fixed_length > 0:
            max_length = args.fixed_length
        else:
            max_length = set_max_length(batch[:, 1:], eos_idx, args)

        source = context_batch
        source_ = source.clone()
        source_[source == pad_idx] = 0

        target = context_batch
        prev = None

        continuation_logits = []

        for _ in range(max_length):

            outputs = model(source, past_key_values=prev)
            logits, prev = outputs.logits, outputs.past_key_values
            logits = logits[:, -1, :]

            if args.top_k == 1 and args.top_p == 0.:
                source = logits.argmax(dim=1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                source = F.softmax(filtered_logits, dim=-1).multinomial(num_samples=1)

            target = torch.cat([target, source], dim=1)
            continuation_logits.append(logits)

        continuation_logits = torch.stack(continuation_logits, 1)

        model.train()

        output_tokens = target[:, args.context_length:].contiguous()
        output_mask = output_tokens.ne(pad_idx).float()
        mask = ngram_repeat_mask(output_tokens, args.seq_n_grams, eos_idx).type_as(continuation_logits)
        log_probs = F.log_softmax(continuation_logits, dim=-1)
        log_probs = log_probs.view(-1, log_probs.size(2)).gather(1, output_tokens.view(-1, 1))
        one_minus_probs = torch.clamp(1. - log_probs.exp(), min=1e-20).view(*output_tokens.size())
        loss = -torch.log(one_minus_probs) * mask * output_mask
        loss = loss.sum()

        metrics = {
            'loss': loss.item(),
            'num_tokens': (mask * output_mask).sum().item()
        }

        return loss, metrics


def evaluate(model, data_source, pad_idx, args):
    model.eval()

    total_loss = 0.
    num_tokens = 0

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

    parser = get_unlikelihood_args()
    args = add_args(parser)
    logger, exp_path = setup('Unlikelihood_{}'.format(args.seq_ul_mix), args)

    logger.info('Build model.')
    tokenizer, pad_idx, eos_idx, model, score_mle_model = load_model(args, mode='train')
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

            if torch.rand(1).item() >= args.seq_ul_mix:
                loss, _ = get_mle_loss(model, batch, pad_idx)
            else:
                loss, seq_metrics = get_seq_loss(model, batch, pad_idx, eos_idx, args)
                train_loss += seq_metrics['loss']
                num_tokens += seq_metrics['num_tokens']

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if idx % args.print_every == 0 and idx > 0:
                cur_loss = train_loss / num_tokens
                logger.info('Epoch: %d, Batch: [%d/%d], Loss: %.3f' % (epoch, idx, args.num_batches, cur_loss))

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
