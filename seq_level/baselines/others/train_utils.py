import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from copy import deepcopy
from others.metrics import task_distance


device = torch.device('cuda')


class GPT2Wrapper(GPT2LMHeadModel):
    pass


class SelfTerminatingWrapper(GPT2LMHeadModel):

    @staticmethod
    def st_softmax(logits, eos_idx, epsilon_upper_bound):
        logits_voc = torch.cat([logits[:, :, :eos_idx], logits[:, :, eos_idx + 1:]], dim=-1)
        logits_eos = logits[:, :, eos_idx].unsqueeze(dim=-1)
        betas = torch.clamp((1. - epsilon_upper_bound) * torch.sigmoid(logits_eos), min=1e-20)
        probs_eos_0 = torch.zeros(betas.size(0), 1, 1, device=device)
        probs_eos_t = 1. - torch.exp(torch.cumsum(torch.log(betas), 1))
        probs_eos_s = torch.cat([probs_eos_0, probs_eos_t], 1)[:, :-1, :]
        alpha = torch.clamp(betas * (1. - probs_eos_s), min=1e-20)
        probs_voc = alpha * torch.softmax(logits_voc, -1)
        probs_ts = torch.clamp(torch.cat([probs_voc, probs_eos_t], dim=2), min=1e-20)
        log_probs = torch.log(probs_ts / probs_ts.sum(-1, keepdim=True))
        return log_probs


def load_model(args, mode='train', name=None):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=args.tokenizer_cache_path)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id

    if mode == 'train':
        if args.train_model_path and name is None:
            model = GPT2Wrapper.from_pretrained(
                args.train_model_path,
                pad_token_id=eos_idx,
                cache_dir=args.transformers_cache_path
            )
            model.to(device)
            score_mle_model = deepcopy(model)
            score_mle_model.to(device)
            return tokenizer, pad_idx, eos_idx, model, score_mle_model
        elif name == 'ST':
            model = SelfTerminatingWrapper.from_pretrained(
                args.train_model_path,
                pad_token_id=eos_idx,
                cache_dir=args.transformers_cache_path
            )
            model.to(device)
            return tokenizer, pad_idx, eos_idx, model
        else:
            model = GPT2LMHeadModel.from_pretrained(
                args.model_name,
                pad_token_id=eos_idx,
                cache_dir=args.transformers_cache_path
            )
            model.to(device)
            return tokenizer, pad_idx, eos_idx, model
    elif mode == 'test':
        if name == 'ST':
            model = SelfTerminatingWrapper.from_pretrained(
                args.test_model_path,
                pad_token_id=eos_idx,
                cache_dir=args.transformers_cache_path
            )
        else:
            model = GPT2Wrapper.from_pretrained(
                args.test_model_path,
                pad_token_id=eos_idx,
                cache_dir=args.transformers_cache_path
            )
        model.to(device)
        score_mle_model = GPT2Wrapper.from_pretrained(
            args.score_mle_model_path,
            pad_token_id=eos_idx,
            cache_dir=args.transformers_cache_path
        )
        score_mle_model.to(device)
        return model, score_mle_model
    else:
        raise NotImplementedError('%s does not match any known mode.' % mode)


def get_optimizer(model, args, is_proxy=False):

    weight_decay = args.proxy_weight_decay if is_proxy else args.weight_decay
    lr = args.proxy_lr if is_proxy else args.lr
    adam_epsilon = args.proxy_adam_epsilon if is_proxy else args.adam_epsilon
    warmup_steps = args.proxy_warmup_steps if is_proxy else args.warmup_steps
    num_batches = args.proxy_num_batches if is_proxy else args.num_batches
    max_epochs = args.proxy_max_epochs if is_proxy else args.max_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(
        params=optimizer_grouped_parameters, lr=lr, eps=adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_batches * max_epochs
    )
    return optimizer, scheduler


def get_mle_loss(model, batch, pad_idx):
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
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_.reshape(-1),
        reduction='none'
    )
    loss = loss * target_mask.view(loss.size())
    loss = loss.sum()
    num_tokens = target_mask.sum()

    metrics = {
        'loss': loss.item(),
        'num_tokens': num_tokens.item(),
        'num_docs': batch.size(0)
    }

    return loss, metrics


def wrap_context_batch(batch, args):
    context_list = []
    for sequence in batch:
        if sequence.size(0) < args.context_length:
            continue
        else:
            context_list.append(sequence[:args.context_length])
    if len(context_list) == 0:
        return torch.tensor([], dtype=torch.long)
    else:
        return torch.stack(context_list, dim=0)


def set_max_length(target, eos_idx, args):
    target_ = target.tolist()[0]
    if eos_idx in target_[args.context_length:]:
        max_length = int(args.decoding_len_factor * (target_.index(eos_idx) + 2))
    else:
        max_length = int(args.decoding_len_factor * (len(target_) + 2))
    max_length = max(1, max_length)
    max_length = min(args.decoding_max_length, max_length)
    return max_length


def _parse_decoding(decoding):
    if decoding == 'greedy':
        args = {
            'do_sample': False
        }
    elif 'beam' in decoding:
        b = int(decoding.split('-')[1])
        args = {
            'do_sample': False,
            'num_beams': b,
            'num_return_sequences': 1,
            'early_stopping': True
        }
    elif 'temp' in decoding:
        temp = float(decoding.split('-')[1])
        args = {
            'do_sample': True,
            'temperature': temp
        }
    elif 'topk' in decoding:
        k = int(decoding.split('-')[1])
        args = {
            'do_sample': True,
            'top_k': k
        }
    elif 'topp' in decoding:
        p = float(decoding.split('-')[1])
        args = {
            'do_sample': True,
            'top_p': p
        }
    else:
        raise NotImplementedError('%s does not match any known decoding method.' % decoding)
    return args


def generate_batch(model, tokenizer, batch, args):

    context_batch = wrap_context_batch(batch, args)
    context_batch = context_batch.to(batch.device)

    eos_idx = tokenizer.eos_token_id

    bpe_prefixes = context_batch.tolist()
    txt_prefixes = [tokenizer.decode(bpe) for bpe in bpe_prefixes]
    bpe_decodings = []
    txt_decodings = []

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
            **_parse_decoding(args.deterministic_decoding)
        )
        full_decodings = decodings.clone()

        for idx in range(decodings.size(0)):
            bpe_decoding = decodings[idx].tolist()
            if eos_idx in bpe_decoding:
                bpe_decoding = bpe_decoding[:bpe_decoding.index(eos_idx) + 1]
            txt_decoding = tokenizer.decode(bpe_decoding)
            bpe_decodings.append(bpe_decoding)
            txt_decodings.append(txt_decoding)

        return bpe_prefixes, txt_prefixes, bpe_decodings, txt_decodings, full_decodings


def sample_batch(model, tokenizer, batch, num_samples, args):

    context_batch = wrap_context_batch(batch, args)
    context_batch = torch.repeat_interleave(context_batch, repeats=num_samples, dim=0)
    context_batch = context_batch.to(batch.device)

    eos_idx = tokenizer.eos_token_id

    bpe_prefixes = context_batch.tolist()
    txt_prefixes = [tokenizer.decode(bpe) for bpe in bpe_prefixes]
    bpe_decodings = []
    txt_decodings = []

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
            **_parse_decoding(args.stochastic_decoding)
        )
        full_decodings = decodings.clone()

        for idx in range(decodings.size(0)):
            bpe_decoding = decodings[idx].tolist()
            if eos_idx in bpe_decoding:
                bpe_decoding = bpe_decoding[:bpe_decoding.index(eos_idx) + 1]
            txt_decoding = tokenizer.decode(bpe_decoding)
            bpe_decodings.append(bpe_decoding)
            txt_decodings.append(txt_decoding)

        return bpe_prefixes, txt_prefixes, bpe_decodings, txt_decodings, full_decodings


def trim_batch(batches, context_outputs, context_length, eos_idx):
    context_targets = batches[:, context_length:]
    if not isinstance(context_targets, list):
        context_targets = context_targets.tolist()
    targets_, outputs_ = [], []
    for target, output in zip(context_targets, context_outputs):
        targets_.append(target[:target.index(eos_idx) + 1] if eos_idx in target else target)
        outputs_.append(output[:output.index(eos_idx) + 1] if eos_idx in output else output)
    return targets_, outputs_


def decode_and_distance(model, score_model, tokenizer, batch, args, deterministic, average=False):
    model.eval()

    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id

    # Sampling
    if deterministic is False:

        num_samples = args.num_samples

        if args.include_greedy:
            num_samples = num_samples - 1

        if args.include_target:
            num_samples = num_samples - 1

        _, _, bpe_decoding, _, full_decodings = sample_batch(model, tokenizer, batch, num_samples, args)
        bpe_decodings = [bpe[args.eval_context_length:] for bpe in bpe_decoding]

        # With/without greedy decoding results
        if args.include_greedy:
            _, _, bpe_decoding_greedy, _, full_decodings_greedy = generate_batch(model, tokenizer, batch, args)
            bpe_decodings_greedy = [bpe[args.eval_context_length:] for bpe in bpe_decoding_greedy]

            bpe_decodings_ = []
            for idx in range(batch.size(0)):
                bpe_decodings_.append(bpe_decodings_greedy[idx])
                bpe_decodings_.extend(bpe_decodings[idx * num_samples:(idx + 1) * num_samples])
            bpe_decodings = bpe_decodings_

            max_length = max(full_decodings.size(1), full_decodings_greedy.size(1))
            outputs = torch.zeros(
                (batch.size(0), num_samples + 1, max_length),
                dtype=torch.long,
                device=full_decodings.device
            )
            outputs[:, 0, :full_decodings_greedy.size(1)] = full_decodings_greedy
            full_decodings = full_decodings.view(batch.size(0), num_samples, -1).contiguous()
            outputs[:, 1:, :full_decodings.size(-1)] = full_decodings
            full_decodings = outputs.view(-1, outputs.size(-1)).contiguous()

            num_samples = num_samples + 1

        batches = torch.repeat_interleave(batch, repeats=num_samples, dim=0)
        targets_trim, outputs_trim = trim_batch(batches, bpe_decodings, args.eval_context_length, eos_idx)

        # With/without targets
        if args.include_target:
            targets_trim_, outputs_trim_ = [], []
            for idx in range(batch.size(0)):
                targets_trim_.extend([targets_trim[idx * num_samples]] * args.num_samples)
                outputs_trim_.append(targets_trim[idx * num_samples])
                outputs_trim_.extend(outputs_trim[idx * num_samples:(idx + 1) * num_samples])
            targets_trim = targets_trim_
            outputs_trim = outputs_trim_

            batch_ = batch.clone()
            batch_[batch == pad_idx] = 0
            max_length = max(batch_.size(1), full_decodings.size(1))

            outputs = torch.zeros(
                (batch_.size(0), args.num_samples, max_length),
                dtype=torch.long,
                device=full_decodings.device
            )
            outputs[:, 0, :batch_.size(1)] = batch_
            full_decodings = full_decodings.view(batch_.size(0), args.num_samples - 1, -1).contiguous()
            outputs[:, 1:, :full_decodings.size(-1)] = full_decodings
            full_decodings = outputs.view(-1, outputs.size(-1)).contiguous()

    # Maximization
    else:
        _, _, bpe_decoding, _, full_decodings = generate_batch(model, tokenizer, batch, args)
        bpe_decodings = [bpe[args.eval_context_length:] for bpe in bpe_decoding]
        targets_trim, outputs_trim = trim_batch(batch, bpe_decodings, args.eval_context_length, eos_idx)

    model.train()

    distances = task_distance(targets_trim, outputs_trim, full_decodings, score_model, args.metric, eos_idx, average)

    return targets_trim, outputs_trim, full_decodings, distances
