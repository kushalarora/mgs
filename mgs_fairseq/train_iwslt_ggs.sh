# NOTE: this is an example command, the experiments in the paper were done with `run_slurm.py`
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 fairseq-train --data data-bin/iwslt14.tokenized.de-en \
    --no-epoch-checkpoints \
    --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed \
    --clip-norm 1.0 \
    --optimizer adam \
    --lr-scheduler fixed \
    --warmup-updates 4000     \
    --dropout 0.3 --weight-decay 0.0001    \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1     \
    --max-tokens 4096     \
    --eval-bleu     \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     \
    --eval-bleu-detok moses    \
    --eval-bleu-remove-bpe     \
    --eval-bleu-print-samples     \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --no-progress-bar  --log-format simple --log-interval 10 \
    --task translation_ggs \
    --ggs-noise 1.0 \
    --ggs-beta 1.0 \
    --ggs-metric sentence_bleu \
    --ggs-num-samples 4 \
    --noise-scaling uniform-global \
    --update-freq 1

