import torch
import numpy as np
from collections import defaultdict, Counter
from nltk import ngrams
import os
import json


class GuidedMetrics(object):
    def __init__(self, name):
        self._stats_cache = defaultdict(list)
        self.name = name

    def step(self,
        distance, decoded, target, pad_token, eos_token,
        model_with_grad=None, update_directions=None, log_rhos=None, log_weights=None
    ):
        with torch.no_grad():
            self._stats_cache['distance'].append(distance)
            len_diffs, target_lens, model_lens = len_stats(decoded, target)
            self._stats_cache['len_diff'].extend(len_diffs)
            self._stats_cache['target_lens'].extend(target_lens)
            self._stats_cache['model_lens'].extend(model_lens)
            self._stats_cache['non_term'].extend(nonterm_metrics(decoded, eos_token))
            for k, vs in ngram_metrics(target).items():
                self._stats_cache['target/%s' % k].extend(vs)
            for k, vs in ngram_metrics(decoded).items():
                self._stats_cache['model/%s' % k].extend(vs)
            for k, v in grad_metrics(model_with_grad, update_directions).items():
                self._stats_cache['model/%s' % k].append(v)
            for k, v in weight_metrics(log_rhos, log_weights).items():
                self._stats_cache['model/%s' % k].append(v)

    def normalize(self):
        output = {}
        output['%s/distance' % self.name] = np.mean(self._stats_cache['distance'])
        for key in ['len_diff', 'target_lens', 'model_lens', 'non_term']:
            output['%s/%s' % (self.name, key)] = np.mean(self._stats_cache[key])
        for key in self._stats_cache:
            if ('grad_cosine' in key or
                'rho' in key or
                'tilde_w' in key or
                'pct_repeat' in key or
                'highest' in key or
                'mle_weight' in key):
                output['%s/%s' % (self.name, key)] = np.mean(self._stats_cache[key])
        return output

    def add_generations(self, prefix, model_trim, target_trim, vocab):
        for i in range(len(prefix)):
            self._stats_cache['generations'].append({
                'prefix': [vocab[tok] for tok in prefix[i]],
                'model': [vocab[tok] for tok in model_trim[i]],
                'target': [vocab[tok] for tok in target_trim[i]],
            })

    def save_generations(self, save_dir, train_step):
        generations = self._stats_cache['generations']
        fp = open(os.path.join(save_dir, "model_generations_%d.json" % train_step), "w")
        json.dump({'generations': generations}, fp)

    def reset(self):
        self._stats_cache = defaultdict(list)


def grad_metrics(model_with_grad, update_directions):
    if model_with_grad is None or update_directions is None:
        return {}
    similarities = []
    for name, param in model_with_grad.named_parameters():
        sim = torch.cosine_similarity(
            param.grad.data.view(1, -1),
            update_directions[name].view(1, -1)
        ).item()
        similarities.append(sim)
    avg = np.mean(similarities)
    std = np.std(similarities)
    output = {
        'avg_grad_cosine': avg,
        'std_grad_cosine': std
    }
    return output


def weight_metrics(log_rhos, log_weights):
    if log_rhos is None or log_weights is None:
        return {}
    rhos = torch.exp(log_rhos)
    weights = torch.exp(log_weights)

    # number of times MLE had the highest weight
    argmax = weights.argmax()
    mle_highest = int(((argmax == 0) or (argmax == 2)).item())

    # number of times zero had the highest weight
    zero_highest = int(((argmax == 1) or (argmax == 3)).item())

    mle_weight = weights[0].item() + weights[2].item()
    output = {
        'avg_rho': rhos.mean().item(),
        'std_rho': rhos.std().item(),
        'avg_tilde_w': weights.mean().item(),
        'std_tilde_w': weights.std().item(),
        'zero_dist_highest': zero_highest,
        'mle_dist_highest':mle_highest,
        'mle_weight': mle_weight,
    }
    return output


def len_stats(preds_trim, targets_trim):
    diffs = []
    target_lens = []
    model_lens = []
    for pred, target in zip(preds_trim, targets_trim):
        diff = np.abs(len(pred) - len(target))
        diffs.append(diff)
        model_lens.append(len(pred))
        target_lens.append(len(target))
    return diffs, target_lens, model_lens


def ngram_metrics(sequences_trim):
    stats = defaultdict(list)
    for sequence in sequences_trim:
        for n in [1, 4]:
            if len(sequence) >= n:
                ngs = [ng for ng in ngrams(sequence, n)]
                counter = Counter([ng for ng in ngrams(sequence, n)])
                stats['pct_repeat_%dgrams' % n].append(
                    1.0 - len(counter)/max(len(ngs), 1)
                )
    return stats


def nonterm_metrics(sequences_trim, eos_token):
    nonterm = []
    for sequence in sequences_trim:
        nonterm.append(float(eos_token not in sequence))
    return nonterm