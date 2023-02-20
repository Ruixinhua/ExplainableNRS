# Explainable News Recommender System

## Introduction
This repository contains the code for the papers [A Novel Perspective to Look At Attention: Bi-level Attention-based 
Explainable Topic Modeling for News Classification](https://arxiv.org/pdf/2203.07216.pdf) and [Topic-Centric Explanations for News Recommendation](). 
The implementation is based on Pytorch and includes _news classification_ and _news recommendation_ tasks. We also provide a procedure to 
generate _explainable topic_ for both news classification task and news recommender system.
## Usage

Clone the repository and install the dependencies.
```bash
git clone https://github.com/Ruixinhua/ExplainedNRS
cd ExplainedNRS
pip install -r requirements.txt
```
Download the MIND dataset from [here](https://msnews.github.io/). The dataset is licensed under the 
[Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). We suggest to put the dataset in the 
dataset folder as follows. 
Download the MIND dataset and GloVe embeddings manually if automatically download failed and put it in the dataset 
folder as follows.

```bash
cd dataset
# Download GloVe pre-trained word embedding and preprocess MIND dataset
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip
mkdir MIND && cd MIND
gdown https://drive.google.com/uc?id=1JUq6UzeGVYifpyupjOD11aKZjaG0Elur
unzip small.zip
rm small.zip
cd ../utils
gdown https://drive.google.com/uc?id=1bC4WgcVrDOAmjVu2o2jR-aETGqbLbveI
```
```bash
cd ../../ # back to the root directory
export PYTHONPATH=PYTHONPATH:./:./modules  # set current directory and the module directory as PYTHONPATH
accelerate launch --config_file config.yaml modules/experiment/runner/run_baseline.py --task=RS_BATM --arch_type=BATMRSModel --mind_type=small --news_info=use_all --news_lengths=100 --word_dict_file=MIND_40910.json --ref_data_path=dataset/utils//ref.dtm.npz --topic_evaluation_method=fast_eval,w2v_sim 
```
## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Reference news recommendation repository by _yusanshi_, see <https://github.com/yusanshi/news-recommendation> 