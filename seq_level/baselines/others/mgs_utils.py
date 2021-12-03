import torch
import torch.nn as nn
from copy import deepcopy
from others.train_utils import get_mle_loss, decode_and_distance


def get_mle_gradient(model, batch, pad_idx, clip_grad_norm=1.0):
    model.zero_grad()
    loss, batch_metrics = get_mle_loss(model, batch, pad_idx)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    return model, batch_metrics


def perturb(model, model_with_grad, score_mle_model, tokenizer, batch, args):

    perturbed_models = []
    log_q_dists = []
    distances = []

    for idx in range(args.num_directions):

        model_ = deepcopy(model)

        eps_eps = 0.
        eps_nab = 0.
        nab_nab = 0.

        for param, (name, param_with_grad) in zip(model_.parameters(), model_with_grad.named_parameters()):

            gradient = -param_with_grad.grad.data

            if args.noise_scale == 'uniform':
                noise_ = args.noise * torch.randn_like(param.data) * (gradient.abs().sum() / gradient.numel())
            else:
                noise_ = args.noise * torch.randn_like(param.data)

            # The mixture of Gaussian: (1 - \pi) * N(mle_gradient, sigma^{2} I) + \pi * N(0, sigma^{2} I)
            # 0, 1: MLE gradient + epsilon
            # 2, 3: random gradient
            if args.zero_dist:
                epsilon = noise_
            elif args.mle_dist:
                epsilon = noise_ + gradient
            else:
                if idx % 2 == 0:
                    epsilon = noise_ + gradient
                else:
                    epsilon = noise_

            param.data = param.data + epsilon

            eps_eps += (epsilon.data.view(-1) * epsilon.data.view(-1)).sum()
            eps_nab += (epsilon.data.view(-1) * gradient.view(-1)).sum()
            nab_nab += (gradient.view(-1) * gradient.view(-1)).sum()

        # Compute the unnormalized probability distribution q(\nabla | \theta)
        if args.zero_dist:
            q_dist = torch.exp(-0.5 * eps_eps)
        elif args.mle_dist:
            q_dist = torch.exp(-0.5 * (eps_eps - 2 * eps_nab + nab_nab))
        else:
            q_dist = 0.5 * torch.exp(-0.5 * eps_eps) + 0.5 * torch.exp(-0.5 * (eps_eps - 2 * eps_nab + nab_nab))

        # Decode with the #num_directions perturbed modules and compute the sequence-level evaluation score
        _, _, _, per_distances = decode_and_distance(
            model_, score_mle_model, tokenizer, batch, args, deterministic=True, average=True
        )
        if args.metric != 'lm':
            per_distances = torch.as_tensor(per_distances).to(batch.device)

        perturbed_models.append(model_)
        log_q_dists.append(q_dist.log())
        distances.append(per_distances)

    log_q_dists = torch.stack(log_q_dists)
    distances = torch.stack(distances)

    return perturbed_models, log_q_dists, distances


def heuristic_perturb(model, model_with_grad, score_mle_model, tokenizer, batch, cur_distances, args):

    perturbed_models = []
    log_q_dists = []
    distances = []

    for idx in range(args.num_directions):

        while True:

            model_ = deepcopy(model)

            eps_eps = 0.
            eps_nab = 0.
            nab_nab = 0.

            for param, (name, param_with_grad) in zip(model_.parameters(), model_with_grad.named_parameters()):

                gradient = -param_with_grad.grad.data

                if args.noise_scale == 'uniform':
                    noise_ = args.noise * torch.randn_like(param.data) * (gradient.abs().sum() / gradient.numel())
                else:
                    noise_ = args.noise * torch.randn_like(param.data)

                if args.zero_dist:
                    epsilon = noise_
                elif args.mle_dist:
                    epsilon = noise_ + gradient
                else:
                    if idx % 2 == 0:
                        epsilon = noise_ + gradient
                    else:
                        epsilon = noise_

                param.data = param.data + epsilon

                eps_eps += (epsilon.data.view(-1) * epsilon.data.view(-1)).sum()
                eps_nab += (epsilon.data.view(-1) * gradient.view(-1)).sum()
                nab_nab += (gradient.view(-1) * gradient.view(-1)).sum()

            _, _, _, per_distances = decode_and_distance(
                model_, score_mle_model, tokenizer, batch, args, deterministic=True, average=True
            )
            if args.metric != 'lm':
                per_distances = torch.as_tensor(per_distances).to(batch.device)

            # Allow MLE gradient to pass it, as it is hard to find MLE-based gradients that improve c(\theta)
            if per_distances - cur_distances.mean() < 0 or idx % 2 == 0:

                if args.zero_dist:
                    q_dist = torch.exp(-0.5 * eps_eps)
                elif args.mle_dist:
                    q_dist = torch.exp(-0.5 * (eps_eps - 2 * eps_nab + nab_nab))
                else:
                    q_dist = 0.5 * torch.exp(-0.5 * eps_eps) + 0.5 * torch.exp(-0.5 * (eps_eps - 2 * eps_nab + nab_nab))

                perturbed_models.append(model_)
                log_q_dists.append(q_dist.log())
                distances.append(per_distances)

                break

    log_q_dists = torch.stack(log_q_dists)
    distances = torch.stack(distances)

    return perturbed_models, log_q_dists, distances


def compute_weights(distance, perturbed_distances, log_q_dists, beta):
    differences = torch.clamp(beta * (distance - perturbed_distances), max=1e16)
    log_weights = torch.log_softmax(differences - log_q_dists, 0)
    return log_weights.exp()


def get_weighted_directions(model, perturbed_models, weights):
    updated_directions = {}
    for name, param in model.named_parameters():
        epsilons = []
        for idx, model_ in enumerate(perturbed_models):
            epsilon = (model_.state_dict()[name] - param).data
            epsilon = weights[idx] * epsilon
            epsilons.append(epsilon)
        final_epsilon = torch.stack(epsilons, 0).sum(0)
        updated_directions[name] = final_epsilon.data
    return updated_directions


def gradient_metrics(model_with_grad, updated_directions):
    similarities = []
    for name, param in model_with_grad.named_parameters():
        similarities.append(
            torch.cosine_similarity(param.grad.data.view(1, -1), -updated_directions[name].view(1, -1))
        )
    similarities = torch.stack(similarities)
    metrics = {
        'avg_grad_cosine': torch.mean(similarities).item(),
        'std_grad_cosine': torch.std(similarities).item()
    }
    return metrics


def weight_metrics(weights):
    argmax = torch.argmax(weights).item()
    is_mle = int(argmax % 2 == 0)
    is_zero = int(argmax % 2 == 1)
    mle_weight = torch.sum(weights[0::2]).item()
    zero_weight = torch.sum(weights[1::2]).item()
    metrics = {
        'is_mle_dist': is_mle,
        'is_zero_dist': is_zero,
        'mle_weight': mle_weight,
        'zero_weight': zero_weight
    }
    return metrics


def get_sparse_weights(weight, topk, eps=10e-8):
    topk += 1
    time_step = weight.size(1)
    if time_step <= topk:
        return weight
    else:
        delta = torch.topk(weight, topk, dim=1)[0][:, -1] + eps
        delta = delta.reshape((delta.size(0), 1))
    weight_w = weight - delta.repeat(1, time_step)
    weight_w = torch.clamp(weight_w, min=0)
    weight_w_sum = torch.sum(weight_w, dim=1, keepdim=True) + eps
    normalized_weight_w = weight_w / weight_w_sum.repeat(1, time_step)
    return normalized_weight_w
