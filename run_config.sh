set -x
wandb=${wandb:="true"}
debug=${debug:="false"}

if [[ ${run_type} =~ mle.* ]];
then
  loss="mle"
  wandb_run="mle-original"

elif [[ "${run_type}" =~ ggs.* ]];
then
  loss="ggs"
  wandb_run="ggs-original"

  if [[ "${run_type}" =~ .*_img.* ]];
  then
    include_mle_grad="true"
    wandb_run="ggs-incl-mle-grad"
  fi

  if [[ "${run_type}" =~ ggs_eff.* ]];
  then
    wandb_run="ggs-efficient"
    efficient="true"
    save_agg_data="true"
    save_score_network="true"
    
    if [ -n "${agg_size}" ] && [ "${agg_size}" != "" ];
    then
      wandb_run+="-${agg_size}"
    fi
    
    if [[ "${run_type}" =~ .*use_agg_data.* ]];
    then
      agg_size=${agg_size:=4000}
      wandb_run+="-use-aggregated-${agg_size}"
      use_agg_data="true"
      agg_data=${agg_data:="datasets/"${agg_size}"_buffer.pkl"}
    fi

    if [[ "${run_type}" =~ .*use_score_network.* ]];
    then
      agg_size=${agg_size:=4000}
      wandb_run+="-use-score-function"
      use_score_network="true"
      score_network=${score_network:="datasets/"${agg_size}"_score_network.pkl"}
    fi

    if [[ "${run_type}" =~ .*_learned.* ]];
    then
      wandb_run+="-learned"
      use_learned_score_function="true"
    fi

    if [[ "${run_type}" =~ .*only_train_score_network.* ]]
    then
      wandb_run+="-only-train-score-network"
      only_train_score_network="true"
    fi
  fi
fi

if [ -n "${wandb_suffix}" ]; then
  wandb_run+="-${wandb_suffix}"
fi

if [[ ${run_type} =~ .*_debug ]];
then
  wandb="false"
  debug="true"
fi

echo "WANDB_RUN_NAME=${wandb_run}"