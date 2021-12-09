import torch
import editdistance
import numpy as np
from collections import defaultdict
from collections import Counter
from nltk import ngrams


class GenerationMetrics(object):
    def __init__(self):

        self._stats_cache = defaultdict(list)

    def step(self, targets, outputs, full_decodings, score_model, eos_idx):

        score_model.eval()
        with torch.no_grad():
            log_probs = torch.log_softmax(score_model(full_decodings)[0], -1)
            log_probs = log_probs[:, :-1, :].gather(2, full_decodings[:, 1:].unsqueeze(-1)).squeeze(-1)
            seq_masks = (full_decodings[:, 1:].eq(eos_idx).cumsum(1).cumsum(1) <= 1).float()
            lm_scores = -(log_probs * seq_masks).sum(1)
            self._stats_cache['lm'].extend(lm_scores.tolist())

        for target, output in zip(targets, outputs):

            # Edit distance
            edit_dist = editdistance.eval(target, output)
            edit_dist = min(edit_dist / len(target), 1.0)
            self._stats_cache['edit'].append(edit_dist)

            # Non-termination
            self._stats_cache['nonterm'].append(float(eos_idx not in output))

            # Average lengths
            self._stats_cache['length_diffs'].append(np.abs(len(target) - len(output)))
            self._stats_cache['target/lengths'].append(len(target))
            self._stats_cache['output/lengths'].append(len(output))

            # Repetition-n
            for n in range(1, 5):

                if len(target) > n:
                    target_ngs = [ng for ng in ngrams(target, n)]
                    target_cnt = Counter(target_ngs)
                    self._stats_cache['target/repetition-%d' % n].append(
                        1.0 - len(target_cnt) / max(len(target_ngs), 1)
                    )

                if len(output) > n:
                    output_ngs = [ng for ng in ngrams(output, n)]
                    output_cnt = Counter(output_ngs)
                    self._stats_cache['output/repetition-%d' % n].append(
                        1.0 - len(output_cnt) / max(len(output_ngs), 1)
                    )

    def normalize(self, prefix='train'):
        outputs = {}
        for key in self._stats_cache:
            outputs['%s/%s' % (prefix, key)] = np.mean(self._stats_cache[key])
        return outputs


def task_distance(targets, outputs, full_decodings, score_model, metric, eos_idx=None, average=True):
    if metric == 'edit':
        edits = []
        for target, output in zip(targets, outputs):
            edit_dist = editdistance.eval(target, output)
            edit_dist = min(edit_dist / len(target), 1.0)
            edits.append(edit_dist)
        if average:
            distance = sum(edits) / len(edits)
        else:
            distance = torch.tensor(edits, dtype=torch.float, device=full_decodings.device)

    elif metric == 'nonterm':
        nonterms = [float(eos_idx not in output) for target, output in zip(targets, outputs)]
        if average:
            distance = sum(nonterms) / len(nonterms)
        else:
            distance = torch.tensor(nonterms, dtype=torch.float, device=full_decodings.device)

    elif metric == 'repeat-n':
        n = int(metric.split('-')[1])
        repetitions = []
        for output in outputs:
            if len(output) >= n:
                output_ngs = [ng for ng in ngrams(output, n)]
                output_cnt = Counter(output_ngs)
                repetitions.append(1.0 - len(output_cnt) / max(len(output_ngs), 1))
        if average:
            distance = np.mean(repetitions)
        else:
            distance = torch.tensor(repetitions, dtype=torch.float, device=full_decodings.device)

    elif metric == 'lm':
        score_model.eval()
        with torch.no_grad():
            log_probs = torch.log_softmax(score_model(full_decodings)[0], -1)
            log_probs = log_probs[:, :-1, :].gather(2, full_decodings[:, 1:].unsqueeze(-1)).squeeze(-1)
            seq_masks = (full_decodings[:, 1:].eq(eos_idx).cumsum(1).cumsum(1) <= 1).float()
            lm_scores = -(log_probs * seq_masks).sum(1)
        if average:
            distance = lm_scores.mean()
        else:
            distance = lm_scores

    else:
        raise NotImplementedError('%s does not match any known evaluation metric.' % metric)

    return distance
