#!/bin/bash -l
# run from current directory
#cd $SLURM_SUBMIT_DIR
nvidia-smi
export PYTHONPATH=PYTHONPATH:./:./modules

# General settings
task=RS_BATM
ds_id=MIND_40910
tensorboard=true
tensorboard_dir=${work_dir}/saved/tensorboard/base
add_entropy_dir=true
gpu_num=1

# Model settings
news_encoder_name=base  # base,multi_view
user_encoder_name=base  # base,gru
arch_type=BATMRSModel
act_layer=tanh          # relu, tanh, prelu


# Training settings
learning_rate=1e-4
dropout_rate=0.2
batch_size=32
epochs=100
early_stop=3
step_size=2
seeds=42,2020,2021,2023,3407  # by default seeds=42,2020,2021,25,4
show_entropy=true   # calculate entropy during training: true/false

# Evaluation settings
valid_method=fast_evaluation  # fast_evaluation/slow_evaluation
find_unused_parameters=false  # true/false
topic_evaluation_method=fast_eval,w2v_sim  # slow_eval,NPMI,W2V,fast_eval
coherence_method=u_mass,c_v,c_uci,c_npmi   # u_mass,c_v,c_uci,c_npmi
evaluate_topic_by_epoch=true
experiment_name=RS_BATM_${subset_type}_${news_encoder_name}_${user_encoder_name}_${topic_variant}
saved_filename=$experiment_name

# Data settings
dataset_name=MIND
news_info=use_all
news_lengths=100
tokenized_method=keep_all  # use_tokenize/keep_all
impression_batch_size=32    # default 1024, too large for 500 topics model
news_batch_size=128
user_history_connect=stack  # concat,stack
ref_data_path=${work_dir}/dataset/utils/wiki.dtm.npz
slow_ref_data_path=${work_dir}/dataset/data/MIND_tokenized.csv
glove_path=${work_dir}/dataset/glove/glove.840B.300d.txt

if [ ${gpu_num} -eq 1 ]
then
  config_file=single.yaml
  export CUDA_VISIBLE_DEVICES=0
else
  config_file=config.yaml
  export CUDA_VISIBLE_DEVICES=0,1
fi

accelerate launch --config_file "$config_file" modules/experiment/runner/run_baseline.py --task="$task" \
--arch_type="$arch_type" --news_encoder_name="$news_encoder_name" --subset_type="$subset_type" \
--topic_variant="$topic_variant" --learning_rate="$learning_rate" --dropout_rate="$dropout_rate" --batch_size="$batch_size" \
--epochs="$epochs" --seeds="$seeds" --dataset_name="$dataset_name" --early_stop="$early_stop" --step_size="$step_size" \
--tokenized_method="$tokenized_method" --evaluate_topic_by_epoch="$evaluate_topic_by_epoch" --head_dim="$head_dim" \
--glove_path="$glove_path" --saved_filename="$saved_filename" --news_info="$news_info" --news_lengths="$news_lengths" \
--saved_dir="${work_dir}/saved" --data_dir="${work_dir}/dataset" --embed_file="${ds_id}.npy" --word_dict_file="${ds_id}.json" \
--arch_attr=head_num --values="$topic_nums" --act_layer="$act_layer" --news_batch_size="$news_batch_size" \
--impression_batch_size="$impression_batch_size" --user_encoder_name="$user_encoder_name" --add_entropy_dir="$add_entropy_dir" \
--ref_data_path="$ref_data_path" --topic_evaluation_method="$topic_evaluation_method" --show_entropy="$show_entropy" \
--alpha="$alpha" --beta="$beta" --valid_method="$valid_method" --user_history_connect="$user_history_connect" \
--find_unused_parameters="$find_unused_parameters" --experiment_name="$experiment_name" --jobid="$SLURM_JOB_ID" \
--coherence_method="$coherence_method" --slow_ref_data_path="$slow_ref_data_path" --gate_type="$gate_type" \
--tensorboard="$tensorboard" --tensorboard_dir="$tensorboard_dir" --entropy_mode="$entropy_mode" \


for JOB in $(jobs -p); do
    wait "${JOB}"
done

