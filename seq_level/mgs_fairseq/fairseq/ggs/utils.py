import editdistance
import torch
import numpy as np
from copy import deepcopy

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score


def task_distance(preds_trim, targets_trim, kind='edit', eos_id=None, bleu_smoothing='method4'):
    if kind == 'edit':
        edits = []
        for pred, target in zip(preds_trim, targets_trim):
            edit_dist = editdistance.eval(target, pred)
            edit_dist = edit_dist / float(len(target))
            edits.append(edit_dist)
        distance = sum(edits) / len(edits)
    elif kind == 'sentence_bleu':
        bleus = []
        for pred, target in zip(preds_trim, targets_trim):
            smoothingf = getattr(SmoothingFunction(), bleu_smoothing)
            bleu = sentence_bleu([target], pred, smoothing_function=smoothingf)
            bleus.append(bleu)
        distance = 1.0 - (sum(bleus) / len(bleus))
    elif kind == 'bleu':
        smoothingf = getattr(SmoothingFunction(), bleu_smoothing)
        target_trim_ = [[x] for x in targets_trim]
        bleu = corpus_bleu(target_trim_, preds_trim, smoothing_function=smoothingf)
        distance = 1 - bleu
    elif kind == 'bleu':
        smoothingf = getattr(SmoothingFunction(), bleu_smoothing)
        target_trim_ = [[x] for x in targets_trim]
        bleu = corpus_bleu(target_trim_, preds_trim, smoothing_function=smoothingf)
        distance = 1 - bleu
    elif kind == 'len_diff':
        diffs = []
        for pred, target in zip(preds_trim, targets_trim):
            diff = np.abs(len(pred) - len(target))
            diff = diff / max(len(pred), len(target))
            diffs.append(diff)
        distance = sum(diffs) / len(diffs)
    elif kind == 'nonterm':
        diffs = []
        for pred, target in zip(preds_trim, targets_trim):
            if eos_id in pred:
                diffs.append(0.0)
            else:
                diffs.append(1.0)
        distance = sum(diffs) / len(diffs)
    elif kind == 'meteor':
        scores = []
        for pred, target in zip(preds_trim, targets_trim):
            score = single_meteor_score(' '.join(target), ' '.join(pred))
            scores.append(score)
        distance = 1.0 - np.mean(scores)
    else:
        raise NotImplementedError(kind)
    return distance


def trim(preds, targets, eos_id, pad_id):
    preds_trim = []
    targets_trim = []
    if not isinstance(preds, list):
        preds = preds.tolist()
    if not isinstance(targets, list):
        targets = targets.tolist()

    for pred, target in zip(preds, targets):
        if eos_id in pred:
            pred = pred[:pred.index(eos_id)+1]
        elif pad_id in pred:
            pred = pred[:pred.index(pad_id)+1]

        if eos_id in target:
            target = target[:target.index(eos_id)+1]
        elif pad_id in target:
            target = target[:target.index(pad_id)+1]
        preds_trim.append(pred)
        targets_trim.append(target)
    return preds_trim, targets_trim


def perturb_mixture(
    model, model_with_grad, num_samples, noise, no_mle, keep_grad, keep_zero,
    noise_scaling='constant',
    last_layer_only=False,
):
    models = []
    log_rhos = []

    for i in range(num_samples):
        model_ = deepcopy(model)
        eps_eps = 0
        eps_nabla = 0
        nabla_nabla = 0
        if noise_scaling == 'uniform-global':
            numer = 0
            denom = 0
            for name, param_with_grad in model_with_grad.named_parameters():
                if last_layer_only and ('transformer.h.11' not in name):
                    continue
                numer += param_with_grad.grad.data.abs().sum()
                denom += param_with_grad.grad.data.numel()


        for param, (name, param_with_grad) in zip(model_.parameters(),
                                                  model_with_grad.named_parameters()):
            if last_layer_only:
                if 'decoder.layers.5' not in name:
                    continue
            g = -param_with_grad.grad.data
            # Generate the noise
            if noise_scaling == 'grad':
                noise_ = noise * torch.randn_like(param.data) * g.abs()
            elif noise_scaling == 'uniform':
                noise_ = noise * torch.randn_like(param.data) * (g.abs().sum() / g.numel())
            elif noise_scaling == 'uniform-global':
                noise_ = noise * torch.randn_like(param.data) * (numer / denom)
            else:
                noise_ = noise * torch.randn_like(param.data)
            # Choose the mixture component (assume 0.5 mixture proportion)
            if i % 2 == 0:
                epsilon = g + noise_
            else:
                epsilon = noise_

            param.data = param.data + epsilon

            eps_eps += (epsilon.data.view(-1)*epsilon.data.view(-1)).sum()
            eps_nabla += (g.view(-1)*epsilon.data.view(-1)).sum()
            nabla_nabla += (g.view(-1)*g.view(-1)).sum()

        q = (0.5*torch.exp(-0.5*eps_eps) + 0.5*torch.exp(-0.5*eps_eps + eps_nabla+ - 0.5*nabla_nabla))
        log_rhos.append(torch.log(q))
        models.append(model_)
    log_rhos = torch.stack(log_rhos).cpu()
    return models, log_rhos


def parameter_weighted_average(model, perturbed_models, log_rhos, log_weights, use_argmax):
    update_directions = {}
    for name, param in model.named_parameters():
        epsilons = []
        if use_argmax:
            i = np.argmax(log_weights)
            epsilon = (perturbed_models[i].state_dict()[name] - param).data
            averaged = torch.exp(log_rhos[i]) * epsilon
        else:
            for i, model_ in enumerate(perturbed_models):
                epsilon = (model_.state_dict()[name] - param).data
                epsilon = torch.exp(log_rhos[i] + log_weights[i]) * epsilon
                epsilons.append(epsilon)
            averaged = torch.stack(epsilons, 0).sum(0)
        update_directions[name] = averaged.data
    return update_directions


def compute_weight(distance, perturbed_distances, log_rhos, beta):
    q = np.exp(-distance)
    qps = [np.exp(-d) for d in perturbed_distances]
    ws = torch.tensor([beta * (np.log(qp) - np.log(q))
                          for qp in qps]).clamp(max=1e16)
    ws = ws - log_rhos
    log_ws = torch.log_softmax(ws, 0)
    return log_ws


def set_update(model, update_directions, optimizer, max_grad_norm, sample_size):
    for name, param in model.named_parameters():
        if param.grad is None:
            param.grad = -update_directions[name]
        else:
            param.grad += -update_directions[name]

