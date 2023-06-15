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

# Model settings
news_encoder_name=base  # base,multi_view
user_encoder_name=base  # base,gru
arch_type=BATMRSModel
act_layer=tanh          # relu, tanh, prelu


# Training settings
learning_rate=0.0005
dropout_rate=0.2
batch_size=32
epochs=100
early_stop=3
step_size=2
seeds=42,  # by default seeds=42,2020,2021,2023,3407
show_entropy=true   # calculate entropy during training: true/false

# Evaluation settings
valid_method=fast_evaluation  # fast_evaluation/slow_evaluation
find_unused_parameters=false  # true/false
topic_evaluation_method=fast_npmi,w2v_sim  # slow_eval,NPMI,W2V,fast_npmi
coherence_method=u_mass,c_v,c_uci,c_npmi   # u_mass,c_v,c_uci,c_npmi
evaluate_topic_by_epoch=true
experiment_name=RS_BATM_${subset_type}_${news_encoder_name}_${user_encoder_name}_${topic_variant}
saved_filename=$experiment_name

# Data settings
dataset_name=MIND
news_info=use_all
news_lengths=100
tokenized_method=keep_all  # use_tokenize/keep_all
impression_batch_size=4    # default 1024, too large for 500 topics model
news_batch_size=128
user_history_connect=stack  # concat,stack
ref_data_path=${work_dir}/dataset/utils/wiki.dtm.npz
slow_ref_data_path=${work_dir}/dataset/data/MIND_tokenized.csv
glove_path=${work_dir}/dataset/glove/glove.840B.300d.txt
