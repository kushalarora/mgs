#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dong.qian@mila.quebec
#SBATCH -o ./log/slurm-%j.out

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate torch181

# 3. Copy your dataset on the compute node
rsync -avz ../datasets/wikitext103_raw_gpt2bpe.pkl ${SLURM_TMPDIR}


DATA_PATH=${data_path:=${SLURM_TMPDIR}/wikitext103_raw_gpt2bpe.pkl}
TRAIN_MODEL_PATH=${train_model_path:=./mle/MLE_seed_$2/}
SCORE_MLE_MODEL_PATH=${score_mle_model_path:=./mle/MLE_seed_$2/}


if [ $1 = mle ];
then
	python train_mle.py --data_path=${DATA_PATH} --seed=$2 --test_model_path=./mle/MLE_seed_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH}
elif [ $1 = pg ];
then
	python train_pg.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --test_model_path=./PG_$3_$4_$5_$2/ --metric=$3 --normalized_distance=$4 --pg_mle_mix=$5
elif [ $1 = gold ];
then
	python train_gold.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./GOLD_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH}
elif [ $1 = mrt ];
then
	python train_mrt.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./MRT_$3_$4_$5_$6_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --metric=$3 --include_target=$4 --normalized_distance=$5 --mrt_mle_mix=$6
elif [ $1 = hinge ];
then
	python train_hinge.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./HG_$3_$4_$5_$6_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --metric=$3 --criterion=$4 --normalized_distance=$5 --hg_mle_mix=$6
elif [ $1 = mgs ];
then
	python train_mgs.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./MGS_$3_$4_$5_$6_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --metric=$3 --num_directions=$4 --zero_dist=$5 --mle_dist=$6 --wandb --wandb_run_name=$7
elif [ $1 = mgsv ];
then
        python train_mgsv.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./MGSv_$3_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --metric=$3
elif [ $1 = es ];
then
	python train_es.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./ES_$3_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --metric=$3
elif [ $1 = unlikelihood ];
then
	python train_unlikelihood.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./Unlikelihood_$3_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --seq_ul_mix=$3
elif [ $1 = st ];
then
        python train_st.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./ST_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH}
elif [ $1 = ours ];
then
        python train_ours.py --data_path=${DATA_PATH} --seed=$2 --train_model_path=${TRAIN_MODEL_PATH} --test_model_path=./OURS_$3_$2/ --score_mle_model_path=${SCORE_MLE_MODEL_PATH} --metric=$3
else
	echo "Input Is Error."
fi

