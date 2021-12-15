from fairseq.tasks.translation import TranslationTask
from fairseq import metrics
import logging
from copy import deepcopy
import torch

from fairseq import bleu, utils
from fairseq.tasks import register_task
import fairseq.ggs.utils as ggs_utils
from fairseq.ggs.sequence_generator import SequenceGenerator
from fairseq.ggs.metrics import GuidedMetrics
import numpy as np
# from sacrebleu import TOKENIZERS

# DEFAULT_TOKENIZER = '13a'

logger = logging.getLogger(__name__)

@register_task('translation_ggs')
class TranslationGGSTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        # fmt: off
        parser.add_argument("--ggs-noise", type=float, default=0.1)
        parser.add_argument("--ggs-beta", type=float, default=100.0)
        parser.add_argument("--max-grad-norm", type=float, default=1.0)
        parser.add_argument("--ggs-num-samples", type=int, default=2)
        parser.add_argument("--decode-len-multiplier", type=int, default=1.3)
        parser.add_argument("--train-decoder", choices=['greedy', 'beam'], default='greedy')
        parser.add_argument("--val-metric", choices=['ppl', 'distance'], default='distance')
        parser.add_argument(
            "--ggs-metric",
            choices=['edit', 'bleu', 'sentence_bleu', 'len_diff', 'nonterm', 'meteor'],
            default='edit'
        )
        parser.add_argument(
            "--bleu-smoothing",
            choices=['method%d' % i for i in range(1, 8)],
            default='method2'
        )
        parser.add_argument(
            "--no-mle", action='store_true',
            help="Zero-mean Gaussian baseline."
        )
        parser.add_argument(
            "--keep-grad", type=int, choices=[0, 1], default=0,
            help="Keep the MLE gradient as an additional sample."
        )
        parser.add_argument(
            "--keep-zero", type=int, choices=[0, 1], default=0,
            help="Keep a 'zero gradient' as an additional sample."
        )
        parser.add_argument(
            "--noise-scaling", type=str, choices=['grad', 'uniform', 'constant', 'uniform-global'], default='uniform-global',
        )
        parser.add_argument(
            "--use-argmax", type=int, choices=[0, 1], default=0
        )
        parser.add_argument(
            "--custom-metrics-interval", type=int, default=10
        )
        parser.add_argument(
            "--validate", action='store_true',
        )
        parser.add_argument(
            "--greedy-eval", type=int, choices=[0, 1], default=0,
        )
        parser.add_argument(
            "--last-layer-only", type=int, choices=[0, 1], default=0,
        )
        parser.add_argument(
            "--mixture", type=int, choices=[0, 1], default=1,
        )
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.custom_metrics = {
            'train': GuidedMetrics('train'),
            'valid': GuidedMetrics('valid')
        }
        self._step = 0

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        args = self.args
        # -- Decode with the current model (required for computing the `weights`)
        decodings_curr = self.decode(model, sample)
        distance_curr = self.distance(decodings_curr, self.tgt_dict.eos(), self.args.ggs_metric)

        # -- Obtain MLE gradients
        model_with_grad = deepcopy(model)
        model_with_grad.zero_grad()
        loss, sample_size, logging_output = criterion(model_with_grad, sample)
        loss_ = loss / sample_size
        loss_.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_with_grad.parameters(), args.max_grad_norm)

        # -- Perturb with noise centered at MLE gradients
        perturb = ggs_utils.perturb_mixture
        perturbed_models, log_rhos = perturb(
            model, model_with_grad, args.ggs_num_samples, args.ggs_noise,
            args.no_mle, args.keep_grad, args.keep_zero,
            noise_scaling=args.noise_scaling,
            last_layer_only=args.last_layer_only
        )

        # -- Decode with perturbed models and compute task metric
        distances = []
        for p_model in perturbed_models:
            decodings = self.decode(p_model, sample)
            distance = self.distance(decodings, self.tgt_dict.eos(), self.args.ggs_metric)
            distances.append(distance)

        # -- Compute weights
        log_weights = ggs_utils.compute_weight(distance_curr, distances, log_rhos, args.ggs_beta)
        # -- Compute weighted average of the directions
        update_directions = ggs_utils.parameter_weighted_average(
            model, perturbed_models, log_rhos, log_weights, args.use_argmax
        )

        # -- Set update directions
        ggs_utils.set_update(model, update_directions, optimizer, args.max_grad_norm, sample_size)

        # -- Periodically compute custom metrics
        if self._step % self.args.custom_metrics_interval == 0:
            self.custom_metrics['train'].step(
                distance_curr, decodings_curr['preds'], decodings_curr['targets'],
                self.tgt_dict.pad(), self.tgt_dict.eos(),
                model_with_grad, update_directions, log_rhos, log_weights
            )
        self._step += 1

        return loss, sample_size, logging_output

    def decode(self, model, sample):
        max_length = int(self.args.decode_len_multiplier*sample['target'].size(1))
        model.eval()
        with torch.no_grad():
            if self.args.train_decoder == 'greedy':
                generator = SequenceGenerator(
                    self.tgt_dict,
                    bos_token=self.tgt_dict.eos(),  # fairseq beam search uses eos as bos
                    max_decoding_length=max_length,
                    return_bos=False
                )
                preds = generator.greedy(model, sample)
            else:
                generator = self.build_generator([model], self.args)
                out = generator(sample)
                preds = [o[0]['tokens'].tolist() for o in out]

        model.train()
        preds_trim, targets_trim = ggs_utils.trim(
            preds, sample['target'], generator.eos, generator.pad
        )

        def tokenize(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
            )
            s = self.tokenizer.decode(s)
            tokens = s.rstrip().split()
            # tokens = TOKENIZERS[DEFAULT_TOKENIZER](s.rstrip()).split()
            return tokens

        for i in range(len(preds_trim)):
            hyp = tokenize(torch.tensor(preds_trim[i]))
            ref = tokenize(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            preds_trim[i] = hyp
            targets_trim[i] = ref

        decodings = {
            'preds': preds_trim,
            'targets': targets_trim
        }
        return decodings

    def distance(self, decodings, eos, metric):
        return ggs_utils.task_distance(
            decodings['preds'], decodings['targets'],
            kind=metric,
            eos_id=eos,
            bleu_smoothing=self.args.bleu_smoothing
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        decodings = self.decode(model, sample)
        logging_output['sentence_bleu'] = self.distance(
            decodings, self.tgt_dict.eos(), 'sentence_bleu'
        )
        logging_output['meteor'] = self.distance(
            decodings, self.tgt_dict.eos(), 'meteor'
        )
        logging_output['edit'] = self.distance(
            decodings, self.tgt_dict.eos(), 'edit'
        )

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)
        metrics.log_scalar('sentence_bleu', sum_logs('sentence_bleu'), round=4)
        metrics.log_scalar('meteor', sum_logs('meteor'), round=4)
        metrics.log_scalar('edit', sum_logs('edit'), round=4)
 