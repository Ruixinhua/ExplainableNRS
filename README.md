# Explainable News Recommender System

## Introduction
This repository contains the code for the papers [A Novel Perspective to Look At Attention: Bi-level Attention-based 
Explainable Topic Modeling for News Classification](https://arxiv.org/pdf/2203.07216.pdf) and [Topic-Centric Explanations for News Recommendation](). 
The implementation is based on Pytorch and includes _news classification_ and _news recommendation_ tasks. We also provide a procedure to 
generate _explainable topic_ for both news classification task and news recommender system.
## Usage

Clone the repository and install the dependencies.
```bash
git clone https://github.com/Ruixinhua/ExplainableNRS
cd ExplainableNRS
pip install -r requirements.txt
```
Download the MIND dataset from [here](https://msnews.github.io/). The dataset is licensed under the 
[Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). We suggest to use the following 
commands to download the processed dataset and pre-trained GloVe word embedding.

```bash
cd dataset
# Download GloVe pre-trained word embedding and preprocessed MIND dataset
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip
mkdir MIND && cd MIND
gdown https://drive.google.com/uc?id=1JUq6UzeGVYifpyupjOD11aKZjaG0Elur
unzip small.zip -d small
rm small.zip
cd ../utils
gdown https://drive.google.com/uc?id=1bC4WgcVrDOAmjVu2o2jR-aETGqbLbveI
```
To run the code for training the basic BATMRS model and evaluating its performance, follow these commands:
```bash
cd ../../ # back to the root directory
export PYTHONPATH=PYTHONPATH:./:./modules  # set current directory and the module directory as PYTHONPATH
accelerate launch modules/experiment/runner/run_baseline.py --task=RS_BATM --arch_type=BATMRSModel --mind_type=small --news_info=use_all --news_lengths=100 --word_dict_file=MIND_40910.json --ref_data_path=dataset/utils/ref.dtm.npz --topic_evaluation_method=fast_eval,w2v_sim 
# use default configuration of accelerate to launch the training script; add "--config_file config.yaml" after launch to use the configuration file
# check [accelerate documentation](https://huggingface.co/docs/accelerate/) for more details
```

## Evaluation
The performance of baselines are summarized in the following table. The results can be obtained by running the code with the default configuration:
![Baselines performance](./plots/baselines_performance.png)

| #Topic | variant | test\_group\_auc | test\_mean\_mrr | test\_ndcg\_5 | test\_ndcg\_10 |
|:-------|:--------|:-----------------|:----------------|:--------------|:---------------|
| 10     | base    | 67.43±0.21       | 32.12±0.34      | 35.78±0.32    | 42.04±0.27     |
| 30     | base    | 67.51±0.14       | 32.16±0.22      | 35.88±0.21    | 42.1±0.18      |
| 50     | base    | 67.65±0.22       | 32.36±0.15      | 36.04±0.21    | 42.26±0.19     |
| 70     | base    | 67.56±0.17       | 32.16±0.31      | 35.85±0.33    | 42.1±0.29      |
| 100    | base    | 67.47±0.19       | 32.14±0.26      | 35.9±0.26     | 42.01±0.25     |
| 150    | base    | 67.38±0.39       | 32.24±0.3       | 35.88±0.37    | 42.08±0.34     |
| 200    | base    | 67.1±0.37        | 32.01±0.44      | 35.56±0.54    | 41.85±0.47     |
| 300    | base    | 66.97±0.39       | 31.94±0.28      | 35.42±0.33    | 41.64±0.34     |
| 500    | base    | 67.22±0.37       | 32.15±0.25      | 35.71±0.3     | 41.94±0.22     |

Topic quality evaluation results are shown in the following table with different configurations of the BATM-ATT model in a NR task:

| #Topic | variant | original\_c\_npmi | PP60\_c\_npmi     | original\_w2v\_sim | PP60\_w2v\_sim    |
|:-------|:--------|:------------------|:------------------|:-------------------|:------------------|
| 10     | base    | 0.0852±0.0075     | 0.1179±0.0221     | 0.2579±0.0184      | 0.2743±0.0218     |
| 30     | base    | 0.0735±0.0273     | 0.1039±0.025      | 0.236±0.0268       | 0.2526±0.0272     |
| 50     | base    | 0.0796±0.0162     | 0.1116±0.0161     | 0.2444±0.0125      | 0.2589±0.0183     |
| 70     | base    | 0.0838±0.0261     | 0.1114±0.0311     | 0.2427±0.0259      | 0.256±0.0324      |
| 100    | base    | 0.0681±0.0059     | 0.1002±0.0073     | 0.2199±0.0057      | 0.2364±0.0095     |
| 150    | base    | 0.0872±0.0126     | 0.1172±0.0146     | 0.2536±0.0083      | 0.2659±0.0117     |
| 200    | base    | 0.0988±0.0101     | 0.1248±0.011      | 0.2778±0.0066      | 0.2874±0.0079     |
| 300    | base    | 0.1044±0.0085     | 0.127±0.0101      | 0.2978±0.0048      | 0.2989±0.0062     |
| 500    | base    | **0.114±0.0054**  | **0.1382±0.0057** | **0.3035±0.0078**  | **0.3108±0.0064** |

Topic quality with variational inference model:

| #Topic | variant | original\_c\_npmi | PP60\_c\_npmi     | original\_w2v\_sim | PP60\_w2v\_sim    |
|:-------|:--------|:------------------|:------------------|:-------------------|:------------------|

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Reference news recommendation repository by _yusanshi_, see <https://github.com/yusanshi/news-recommendation> 