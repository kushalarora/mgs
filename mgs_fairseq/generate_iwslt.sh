# This script is used to generate `Bleu` results in the paper.
#!/usr/bin/env bash

# Model paths
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
    fairseq-generate --data /data/iwslt14.tokenized.de-en \
        --path $i \
        --beam 5 \
        --remove-bpe \
        --lenpen 1 \
        --gen-subset $DATA \
        --quiet
done

