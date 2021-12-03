import torch
import numpy as np
import faiss
import math
from sklearn.metrics import auc as compute_area_under_curve
from sklearn.preprocessing import normalize
from transformers import GPT2LMHeadModel
from sklearn.decomposition import PCA
from others.train_utils import get_mle_loss, generate_batch, trim_batch
from others.metrics import GenerationMetrics


def evaluate(model, score_model, tokenizer, data_source, args):
    model.eval()

    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id

    total_loss = 0.
    num_tokens = 0

    gen_metrics = GenerationMetrics()

    with torch.no_grad():

        for _, batch in enumerate(data_source):

            batch = batch.squeeze(0)
            batch = batch.to(model.device)

            if (batch[:, :args.eval_context_length + 1] == pad_idx).sum() > 0:
                continue

            _, batch_metrics = get_mle_loss(model, batch, pad_idx)
            total_loss += batch_metrics['loss']
            num_tokens += batch_metrics['num_tokens']

            _, _, bpe_decoding, _, full_decodings = generate_batch(model, tokenizer, batch, args)
            bpe_decodings = [bpe[args.eval_context_length:] for bpe in bpe_decoding]
            targets_trim, outputs_trim = trim_batch(batch, bpe_decodings, args.eval_context_length, eos_idx)
            gen_metrics.step(targets_trim, outputs_trim, full_decodings, score_model, eos_idx)

        report_loss = total_loss / num_tokens
        report_ppl = np.exp(report_loss)

        metrics = {}
        gen_metrics = gen_metrics.normalize('valid')
        for k, v in gen_metrics.items():
            metrics[k] = v

    return report_loss, report_ppl, metrics


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

            _, batch_metrics = get_mle_loss(model, batch, pad_idx)
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


def compute_mauve(
        tokenizer=None,
        p_tokens=None,
        q_tokens=None,
        num_buckets='auto',
        pca_max_data=-1,
        kmeans_explained_variance=0.9,
        kmeans_num_redos=5,
        kmeans_max_iter=500,
        model_name='gpt2',
        divergence_curve_discretization_size=25,
        mauve_scaling_factor=5
):
    p_features = get_features_from_inputs(tokenizer, p_tokens, model_name)
    q_features = get_features_from_inputs(tokenizer, q_tokens, model_name)

    if num_buckets == 'auto':
        # heuristic: use num_clusters = num_generations / 10
        num_buckets = max(2, int(round(min(p_features.shape[0], q_features.shape[0]) / 10)))
    elif not isinstance(num_buckets, int):
        raise ValueError('The num_buckets is expected to be an integer or "auto".')

    p, q = cluster_features(
        p_features,
        q_features,
        num_clusters=num_buckets,
        norm='l2',
        whiten=False,
        pca_max_data=pca_max_data,
        kmeans_explained_variance=kmeans_explained_variance,
        kmeans_num_redos=kmeans_num_redos,
        kmeans_max_iter=kmeans_max_iter
    )
    mixture_weights = np.linspace(1e-6, 1-1e-6, divergence_curve_discretization_size)
    divergence_curve = get_divergence_curve_for_multinomials(p, q, mixture_weights, mauve_scaling_factor)
    x, y = divergence_curve.T
    idxs1 = np.argsort(x)
    idxs2 = np.argsort(y)
    mauve_score = 0.5 * (compute_area_under_curve(x[idxs1], y[idxs1]) + compute_area_under_curve(y[idxs2], x[idxs2]))
    fi_score = get_frontier_integral(p, q)
    metrics = {
        'p_hist': p,
        'q_hist': q,
        'divergence_curve': divergence_curve,
        'mauve': mauve_score,
        'frontier_integral': fi_score,
        'num_buckets': num_buckets
    }
    return metrics


def get_features_from_inputs(tokenizer, tokenized_texts, model_name, device=torch.device('cuda')):
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        pad_token_id=tokenizer.eos_token_id,
        cache_dir='../models/transformers_cache/'
    )
    model.to(device)
    model.eval()
    with torch.no_grad():
        features = featurize_tokens_from_model(model, tokenized_texts).detach().cpu().numpy()
    return features


def featurize_tokens_from_model(model, batch):
    features = []
    for sequence in batch:
        if isinstance(sequence, list):
            sequence = torch.LongTensor(sequence).unsqueeze(0)
        sequence = sequence.to(model.device)
        outputs = model(input_ids=sequence,
                        past_key_values=None,
                        output_hidden_states=True,
                        return_dict=True)
        hidden = outputs.hidden_states[-1]
        features.append(hidden[:, -1, :].cpu())
    return torch.cat(features)


def cluster_features(
        p, q,
        num_clusters,
        norm='none',
        whiten=True,
        pca_max_data=-1,
        kmeans_explained_variance=0.9,
        kmeans_num_redos=5,
        kmeans_max_iter=500
):
    assert 0 < kmeans_explained_variance < 1
    assert norm in ['none', 'l2', 'l1', None]

    data = np.vstack([q, p])

    if norm in ['l2', 'l1']:
        data = normalize(data, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten)

    if pca_max_data < 0 or pca_max_data >= data.shape[0]:
        pca.fit(data)
    elif 0 < pca_max_data < data.shape[0]:
        ids = np.random.choice(data.shape[0], size=pca_max_data, replace=False)
        pca.fit(data[ids])
    else:
        raise ValueError(f'Invalid argument pca_max_data={pca_max_data} with {data.shape[0]} data points.')

    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= kmeans_explained_variance)

    data1 = pca.transform(data)[:, :idx + 1]
    data1 = data1.astype(np.float32)
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters,
                          niter=kmeans_max_iter, nredo=kmeans_num_redos, update_index=True)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)

    q_labels = labels[:len(q)]
    p_labels = labels[len(q):]

    q_bins = np.histogram(q_labels, bins=num_clusters, range=[0, num_clusters], density=True)[0]
    p_bins = np.histogram(p_labels, bins=num_clusters, range=[0, num_clusters], density=True)[0]

    return p_bins / p_bins.sum(), q_bins / q_bins.sum()


def get_divergence_curve_for_multinomials(p, q, mixture_weights, scaling_factor):
    divergence_curve = [[0, np.inf]]
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0])
    return np.exp(-scaling_factor * np.asarray(divergence_curve))


def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_frontier_integral(p, q, scaling_factor=2):
    total = 0.
    for p1, q1 in zip(p, q):
        if p1 == 0 and q1 == 0:
            pass
        elif p1 == 0:
            total += q1 / 4
        elif q1 == 0:
            total += p1 / 4
        elif abs(p1 - q1 > 1e-8):
            t1 = p1 + q1
            t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
            total += 0.25 * t1 - 0.5 * t2
    return total * scaling_factor
