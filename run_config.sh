set -x
wandb=true
debug=false

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
    
    if [[ "${run_type}" =~ .*_agg.* ]];
    then
      wandb_run+="-aggregated"
      use_agg_data="true"
      agg_data=${agg_data:="datasets/4000_buffer.pkl"}
    fi

    if [[ "${run_type}" =~ .*_score.* ]];
    then
      wandb_run+="-score-function"
      use_score_network="true"
      score_network=${score_network:="datasets/4000_score_network.pkl"}
    fi

    if [[ "${run_type}" =~ .*_learned.* ]];
    then
      wandb_run+="-learned"
      use_learned_score_function="true"
    fi
  fi
fi

if [[ ${run_type} =~ .*_debug ]];
then
  wandb="false"
  debug="true"
fi