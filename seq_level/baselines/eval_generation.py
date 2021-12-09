import torch
import numpy as np
import argparse
import logging
import random
import glob
import os
from torch.utils.data import DataLoader, SequentialSampler
from transformers import GPT2Tokenizer
from collections import defaultdict
from others.train_utils import wrap_context_batch, set_max_length, _parse_decoding, SelfTerminatingWrapper, GPT2Wrapper
from others.eval_utils import compute_mauve
from others.data_utils import load_dataset


device = torch.device('cuda')


def get_generation_args():
    parser = argparse.ArgumentParser(description='Evaluate generation performance.')
    parser.add_argument(
        '--data_path', type=str, default='../datasets/wikitext103_raw_gpt2bpe.pkl'
    )
    parser.add_argument(
        '--chunk_size_train', type=int, default=1024
    )
    parser.add_argument(
        '--chunk_size_valid', type=int, default=1024
    )
    parser.add_argument(
        '--token_limit_train', type=int, default=1024
    )
    parser.add_argument(
        '--token_limit_valid', type=int, default=1024
    )
    parser.add_argument(
        '--context_length', type=int, default=10
    )
    parser.add_argument(
        '--tokenizer_cache_path', type=str, default='../models/tokenizer_cache/'
    )
    parser.add_argument(
        '--model_name', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2'
    )
    parser.add_argument(
        '--transformers_cache_path', type=str, default='../models/transformers_cache/'
    )
    parser.add_argument(
        '--model_path', type=str, default='./mle/'
    )
    parser.add_argument(
        '--eval_context_length', type=int, default=10
    )
    parser.add_argument(
        '--decoding_max_length', type=int, default=500
    )
    parser.add_argument(
        '--decoding_len_factor', type=float, default=1.3
    )
    parser.add_argument(
        '--fixed_length', type=int, default=-1
    )
    args = parser.parse_args()
    return args


def load_model(test_model_path, eos_idx, args):

    string = test_model_path.split('/')[-2].split('_')
    name, seed = string[0], int(string[-1])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if name == 'ST':
        model = SelfTerminatingWrapper.from_pretrained(
            test_model_path,
            pad_token_id=eos_idx,
            cache_dir=args.transformers_cache_path
        )
    else:
        model = GPT2Wrapper.from_pretrained(
            test_model_path,
            pad_token_id=eos_idx,
            cache_dir=args.transformers_cache_path
        )
    model.to(device)

    return model


def evaluate(model, tokenizer, data_source, method, args):
    model.eval()

    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id

    targets_prefixes, outputs_prefixes = [], []

    with torch.no_grad():

        for _, batch in enumerate(data_source):

            batch = batch.squeeze(0)
            batch = batch.to(model.device)

            if (batch[:, :args.eval_context_length + 1] == pad_idx).sum() > 0:
                continue

            context_batch = wrap_context_batch(batch, args)
            context_batch = context_batch.to(batch.device)

            if context_batch.size(0) > 0:

                if args.fixed_length > 0:
                    max_length = args.fixed_length
                    min_length = args.fixed_length
                else:
                    max_length = set_max_length(batch[:, 1:], eos_idx, args)
                    min_length = 10

                decodings = model.generate(
                    input_ids=context_batch,
                    max_length=max_length,
                    min_length=min_length,
                    eos_token_id=eos_idx,
                    **_parse_decoding(method)
                )
                full_decodings = decodings.clone()

            if not isinstance(batch, list):
                batch = batch.tolist()
                full_decodings = full_decodings.tolist()

            for target, output in zip(batch, full_decodings):
                targets_prefixes.append(target[:target.index(eos_idx) + 1] if eos_idx in target else target)
                outputs_prefixes.append(output[:output.index(eos_idx) + 1] if eos_idx in output else output)

        outputs = {
            'bpe_tgt_including_prefixes': targets_prefixes,
            'bpe_out_including_prefixes': outputs_prefixes
        }

    return outputs


def main():
    args = get_generation_args()

    logging.basicConfig(
        filename=os.path.join(args.model_path, 'log.txt'),
        filemode='w',
        format='%(asctime)s - %(levelname)s -  %(message)s',
        datefmt='%Y-%m-%d_%H-%M-%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=args.tokenizer_cache_path)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id

    logger.info('Load data.')
    dataset = load_dataset(pad_idx, args)
    test_dataloader = DataLoader(dataset=dataset['test'], sampler=SequentialSampler(dataset['test']))

    metrics = defaultdict(list)

    for ckpt in sorted(glob.glob(args.model_path + '*/')):

        logger.info('Load model from %s.' % ckpt)

        model = load_model(ckpt, eos_idx, args)

        for method in ['greedy', 'temp-1.0', 'topp-0.95']:

            decodings = evaluate(model, tokenizer, test_dataloader, method, args)

            # Compute the MAUVE
            out = compute_mauve(tokenizer=tokenizer,
                                p_tokens=decodings['bpe_tgt_including_prefixes'],
                                q_tokens=decodings['bpe_out_including_prefixes'],
                                model_name=args.model_name)
            logger.info(out['mauve'])

            for k, v in out.items():
                metrics['%s/%s/%s/%s' % (ckpt, args.model_name, method, k)].append(v)

    logger.info('Evaluation metrics:')
    for k, v in metrics.items():
        logger.info(k, v)


if __name__ == '__main__':
    main()
