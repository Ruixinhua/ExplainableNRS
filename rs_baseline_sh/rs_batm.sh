#!/bin/bash -l
#SBATCH --job-name=RS-base-entropy-wiki-small-fast
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=csgpu
# Request 2 gpus
#SBATCH --gres=gpu:2
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35


# specify the walltime e.g 20 mins
#SBATCH -t 168:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dairui.liu@ucdconnect.ie
# run from current directory
cd $SLURM_SUBMIT_DIR
nvidia-smi
export PYTHONPATH=PYTHONPATH:./:./modules
source $SLURM_SUBMIT_DIR/rs_baseline_sh/setup.sh
# command to use
news_encoder_name=base  # base,multi_view
user_encoder_name=base  # base,gru
user_history_connect=stack  # concat,stack
topic_variant=base_adv  # base,add_dense,raw,variational_topic,topic_embed,base_gate,base_adv
gate_type=close  # close,multiply
beta=1
valid_method=fast_evaluation  # fast_evaluation/slow_evaluation
entropy_mode=static  # dynamic/static
show_entropy=true
alpha=0
add_entropy_dir=true
news_lengths=100
tensorboard_dir=${work_dir}/saved/tensorboard/${topic_variant}
find_unused_parameters=true  # true/false
topic_evaluation_method=fast_eval,w2v_sim   # slow_eval,NPMI,W2V,fast_eval
coherence_method=u_mass,c_v,c_uci,c_npmi  # u_mass,c_v,c_uci,c_npmi
slow_ref_data_path=${work_dir}/dataset/data/MIND_tokenized.csv

gpu_num=2
evaluate_topic_by_epoch=false
tensorboard=true
news_info=use_all
subset_type=small
head_dim=30
act_layer=tanh  # relu, tanh, prelu
lr=0.00005
batch_size=32
impression_batch_size=32  # default 1024, too large for 500 topics model
news_batch_size=256
epochs=100
experiment_name=Win200
#seeds=42,2020,2021,25,4  # by default seeds=42,2020,2021,25,4
seeds=42,2020,2021,25,4
#topic_nums=10,200
topic_nums=30,50,70,100,300,500 # 10,30,50,70,100,150,200,300,500
arch_type=BATMRSModel
tokenized_method=keep_all  # use_tokenize/keep_all
saved_filename=RS_BATM_${subset_type}_${news_encoder_name}_${user_encoder_name}_${topic_variant}_${act_layer}_hd${head_dim}  # for base model
#saved_filename=RS_BATM_${subset_type}_${topic_variant}_${act_layer}  # for raw model
dataset_name=MIND
# MIND_40910 is the original MIND dataset
ds_id=MIND_40910  # MIND_large_42301/MIND_small_31139/MIND15_41684 for word dictionary and embedding file
#ref_data_path=dataset/utils/large.dtm.npz  # dataset/utils/ref.dtm.npz
ref_data_path=${work_dir}/dataset/utils/wiki.dtm.npz

if [ $gpu_num -eq 1 ]
then
  config_file=single.yaml
  export CUDA_VISIBLE_DEVICES=0
else
  config_file=config.yaml
  export CUDA_VISIBLE_DEVICES=0,1
fi
#CUDA_VISIBLE_DEVICES=0
#for no in "1" "0";do
#  CUDA_VISIBLE_DEVICES=$no accelerate launch --config_file config.yaml modules/experiment/runner/run_baseline.py \
accelerate launch --config_file $config_file modules/experiment/runner/run_baseline.py --task=RS_BATM \
--arch_type=$arch_type --news_encoder_name=$news_encoder_name --subset_type=$subset_type --topic_variant=$topic_variant \
--learning_rate=$lr --dropout_rate=0.2 --batch_size=$batch_size --epochs=$epochs --seeds=$seeds \
--dataset_name=$dataset_name --early_stop=5 --step_size=2 --news_info=$news_info --news_lengths=$news_lengths \
--tokenized_method=$tokenized_method --evaluate_topic_by_epoch=$evaluate_topic_by_epoch --head_dim=$head_dim \
--glove_path="${work_dir}"/dataset/glove/glove.840B.300d.txt --saved_filename=$saved_filename \
--saved_dir="${work_dir}"/saved --data_dir="${work_dir}"/dataset --embed_file=$ds_id.npy --word_dict_file=$ds_id.json \
--arch_attr=head_num --values=$topic_nums --act_layer=$act_layer --news_batch_size=$news_batch_size \
--impression_batch_size=$impression_batch_size --user_encoder_name=$user_encoder_name --add_entropy_dir=$add_entropy_dir \
--ref_data_path="$ref_data_path" --topic_evaluation_method=$topic_evaluation_method --show_entropy=$show_entropy \
--alpha=$alpha --beta=$beta --valid_method=$valid_method --user_history_connect=$user_history_connect \
--find_unused_parameters=$find_unused_parameters --experiment_name=$experiment_name --jobid="${SLURM_JOB_ID}" \
--coherence_method=$coherence_method --slow_ref_data_path="${slow_ref_data_path}" --gate_type=$gate_type \
--tensorboard=$tensorboard --tensorboard_dir="${tensorboard_dir}" --entropy_mode=$entropy_mode
#done

for JOB in $(jobs -p); do
    wait "${JOB}"
done

