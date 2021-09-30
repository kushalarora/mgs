#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:16GB:1
##SBATCH --constraint=nvlink
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH -o ./log/slurm-%j.out

# 1. Load the required modules
# module --quiet load anaconda/3

# 2. Load your environment
# conda activate torch181

module load python/3.7
if [ ! -d ${HOME}/envs/mgs ]; then
	python -m venv ${HOME}/envs/mgs;
	source ${HOME}/envs/mgs/bin/activate;
	python setup.py develop;
	deactivate;
fi
source ${HOME}/envs/mgs/bin/activate

# 3. Copy your dataset on the compute node
rsync -avz ./datasets/wikitext103_raw_gpt2bpe.pkl $SLURM_TMPDIR

MODEL_NAME=${model_name:="gpt2"}
loss=${loss:="mle"}
TPORT=${port:=8001}
EXP_NAME=${exp_name:="wikipedia103/"}"_${MODEL_NAME}_${loss}"
OUTPUT_DIR_SUFFIX="${MODEL_NAME}_${loss}_${SLURM_JOB_ID}"
SAVE_BASE_DIR=${save_dir:-"./results/wikipedia103"}
SAVE_DIR=${SAVE_BASE_DIR}/${OUTPUT_DIR_SUFFIX}
if [ -d ${SAVE_DIR} ]; then
	mkdir -p ${SAVE_DIR}
fi 

. ./run_config.sh

cmd="python -u seq_level/gpt2/train.py --dataset-path=$SLURM_TMPDIR/wikitext103_raw_gpt2bpe.pkl"

if [  -n "${wandb}" ] && [ "${wandb}" == "true" ];
then
	cmd+=" --wandb "

	if [ -n "${wandb_run}" ]; 
	then
		cmd+=" --wandb-run-name ${wandb_run} "
	fi

	if [ -n "${wandb_tags}" ]; then
		cmd+=" --wandb-tags ${wandb_tags} "
	fi
fi
if [ ${loss} = mle ];
then
	cmd+=" --loss mle --valid-every 5000 --print-every 100 "
else
	GGS_METRIC=${ggs_metric:="lm"}
	MLE_MIX=${mle_mix:=0.1}
	MODEL_LOAD_DIR=${model_dir:="./mle/default/"}
	cmd+=" --ggs-metric ${GGS_METRIC} --model-load-dir=${MODEL_LOAD_DIR} "

	if [ ${loss} = pg ];
	then
		PG_NORMALIZE_DISTANCE=${pg_dist:=1}
		PG_BASELINE=${baseline:="avg"}
		cmd+="  --loss pg --pg-normalize-distance=${PG_NORMALIZE_DISTANCE} --pg-mle-mix=${MLE_MIX} --pg-baseline ${PG_BASELINE} "

	elif [ ${loss} = mrt ];
	then
		MRT_NORMALIZE_DISTANCE=${mrt_dist:=1}
		MRT_BASELINE=${baseline:="avg"}
		cmd+=" --loss mrt --mrt-normalize-distance=${MRT_NORMALIZE_DISTANCE} --mrt-mle-mix=${MLE_MIX} "
		
	elif [ ${loss} = ggs ];
	then
		MGS_BETA=${mgs_beta:=1.0}
		cmd+=" --loss ggs --ggs-beta=${MGS_BETA}"

        if [ -n "${include_mle_grad}" ] && [ "${include_mle_grad}" == "true" ];
        then
            cmd+=" --include-mle-gradient "
        fi

        if [ -n "${efficient}" ] && [ "${efficient}" == "true" ]; 
        then
            cmd+=" --efficient  --log-scoring-function  --on-device"
						if [ -n "${debug}" ] && [ "${debug}" == "true" ];
						then
							cmd+=' --score-network-epochs 100 --aggregated-data-size 40 --retrain-score-network-every 100 --max-buffer-size 80 --on-device'
							# cmd+=' --score-network-epochs 50 --aggregated-data-size 300 --retrain-score-network-every 200 --max-buffer-size 600 '
						fi

						if [ -n "${use_agg_data}" ] && [ "${use_agg_data}" == "true" ];
						then
							if [ -z "${agg_data}" ];
							then
								echo "To use aggregated data, agg_data filepath is needed."
								exit;
							fi
							cmd+=" --use-saved-aggregated-data --aggregated-data-path ${agg_data} "
						fi

						if [ -n "${save_agg_data}" ] && [ "${save_agg_data}" == "true" ];
						then
							cmd+=" --save-aggregated-data "
						fi

						if [ -n "${use_score_network}" ] && [ "${use_score_network}" == "true" ];
						then
							if [ -z "${score_network}" ];
							then
								echo "To use saved score network, score_network filepath is needed."
								exit;
							fi
							cmd+=" --use-saved-score-network --score-network-file ${score_network} "
						fi

						if [ -n "${save_score_network}" ] && [ "${save_score_network}" == "true" ];
						then
							cmd+=" --save-score-network "
						fi

						if [ -n "${use_learned_score_function}" ] && [ "${use_learned_score_function}" == "true" ];
						then
							cmd+=" --use-learned-scoring-function "
						fi

						if [ -n "${only_train_score_network}" ] && [ "${only_train_score_network}" == "true" ];
						then
							cmd+=" --only-train-score-network "
						fi

						if [ -n "${agg_size}" ];
						then
							max_buffer_size=$(expr ${agg_size} \* 2)
							cmd+=" --aggregated-data-size ${agg_size} --max-buffer-size ${max_buffer_size} "
						fi

						if [ -n "${score_patience}" ];
						then
							cmd+="  --train-score-patience ${score_patience} ";
						fi

						if [ -n "${scorer_type}" ]; then
							cmd+=" --score-network-type ${scorer_type} ";
						fi

						if [ -n "${scorer_nlayers}" ]; then
							cmd+=" --score-network-num-layers ${scorer_nlayers} ";
						fi
						if [ -n "${scorer_lr}" ]; then
							cmd+=" --scorer-lr ${scorer_lr} ";
						fi
        fi
	else
		echo "Input Is Error."
		exit
	fi
fi
TMP_RUN_DIR=${SLURM_TMPDIR}/${OUTPUT_DIR_SUFFIX}

cmd+=" --save-base-dir ${TMP_RUN_DIR}"

pkill -f "port ${TPORT}"
echo "Running Command:"

echo "	$cmd"

tensorboard --logdir ${TMP_RUN_DIR} --port ${TPORT} --host localhost &
# For Tensorboard port forwarding based on https://josephpcohen.com/w/jupyter-notebook-and-hpc-systems/.
ssh -N -R ${TPORT}:localhost:${TPORT} login-4 &


$cmd

if [ -z "${debug}" ]; then
	rsync -avz ${TMP_RUN_DIR} ${SAVE_DIR}
fi
