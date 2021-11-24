# This script is used to compute the greedy metrics in the paper (SBleu, Meteor, Edit)
#!/usr/bin/env bash

declare -a paths=(
                  "/iwslt_mle/checkpoint_best.pt"
                  "/iwslt_ggs_tune____ggs-metric=sentence_bleu/checkpoint_best.pt"
                  "/iwslt_ggs_tune____ggs-metric=meteor/checkpoint_best.pt"
                  "/iwslt_ggs_tune____ggs-metric=edit/checkpoint_best.pt"
                  "/iwslt_ggs_fromscratch/checkpoint_best.pt"
)

DATA=$1  # valid or test

for i in "${paths[@]}"
do
    fairseq-train --data /data/iwslt14.tokenized.de-en \
        --validate \  # <-- this is a custom argument that does validation instead of training
        --no-epoch-checkpoints \
        --arch transformer_iwslt_de_en \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 1.0 \
        --lr-scheduler fixed \
        --warmup-updates 4000     \
        --dropout 0.3 --weight-decay 0.0001    \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1     \
        --max-tokens 4096     \
        --eval-bleu     \
        --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10.0}'     \
        --eval-bleu-detok moses    \
        --eval-bleu-remove-bpe     \
        --eval-bleu-print-samples     \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --no-progress-bar  --log-format simple --log-interval 10 \
        --task translation_ggs \
        --ggs-noise 1.0 \
        --ggs-metric len_diff \
        --restore-file $i \
        --valid-subset $DATA \
        --greedy-eval 1
done
