# Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics
This is the official repository of NAACL 2022 paper "*Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics*".

Paper is available at [Placeholder](https://google.com).

## How to Run

```shell
conda create -n cluster_topic_model python=3.7 -y
conda activate cluster_topic_model
pip install -r requirements.txt
bash run_evaluate.sh
```

### Dataset

We use preproessed dataset from [OCTIS](https://github.com/MIND-Lab/OCTIS#datasets-and-preprocessing). You can choose from `[bbc, m10, 20ng]`. 

### Topic Model

You can run our model `cetopic` or you can also choose a baseline model from `[lda, prodlda, zeroshottm, combinedtm, bertopic]`.

### Word Selecting Method

If you use `cetopic`, you can also choose a word selecting method from `[tfidf_idfi, tfidf_tfi, tfidfi, tfi]`.

### Pretrained Model
You can choose a pretrained model such as `princeton-nlp/unsup-simcse-bert-base-uncased` or `bert-base-uncased` from [SimCSE](https://github.com/princeton-nlp/SimCSE) or [HuggingFace](https://huggingface.co/models).

### Examples

\>> Run `cetopic` on BBC dataset using `tfidf_idfi` word selecting method and unsupervised SimCSE embeddings, the embedding dimensionality will be reduced to 5 and will ouput 5 topics:
```shell
# run_evaluate.sh

DATASET='bbc'
TOPIC_MODEL='cetopic'
WORD_SELECT_METHOD='tfidf_idfi'
PRETRAINED_MODEL='princeton-nlp/unsup-simcse-bert-base-uncased'

python main.py\
    --topic_model ${TOPIC_MODEL}\
    --dataset ${DATASET}\
    --k 5\
    --dim_size 5\
    --word_select_method ${WORD_SELECT_METHOD}\
    --pretrained_model ${PRETRAINED_MODEL}\
    --seed 30
```

\>> Run `bertopic` on 20NewsGroup dataset using BERT embeddings and expect to ouput 50 topics. *Note that BERTopic may not output the exact specified number of topics*:
```shell
# run_evaluate.sh

DATASET='20ng'
TOPIC_MODEL='bertopic'
PRETRAINED_MODEL='bert-base-uncased'

python main.py\
    --topic_model ${TOPIC_MODEL}\
    --dataset ${DATASET}\
    --k 50\
    --pretrained_model ${PRETRAINED_MODEL}\
```

\>> Run `combinedtm` on M10 dataset using RoBERTa embeddings and expect to ouput 75 topics:
```shell
# run_evaluate.sh

DATASET='m10'
TOPIC_MODEL='combinedtm'
PRETRAINED_MODEL='roberta-base'

python main.py\
    --topic_model ${TOPIC_MODEL}\
    --dataset ${DATASET}\
    --k 75\
    --pretrained_model ${PRETRAINED_MODEL}\
```

### Arguments
```
usage: main.py [-h] [--topic_model TOPIC_MODEL] [--dataset DATASET]
               [--pretrained_model PRETRAINED_MODEL] [--k K]
               [--dim_size DIM_SIZE] [--word_select_method WORD_SELECT_METHOD]
               [--seed SEED]

Cluster Contextual Embeddings for Topic Models

optional arguments:
  -h, --help            show this help message and exit
  --topic_model TOPIC_MODEL
                        Topic model to run experiments
  --dataset DATASET     Datasets to run experiments
  --pretrained_model PRETRAINED_MODEL
                        Pretrained language model
  --k K                 Topic number
  --dim_size DIM_SIZE   Embedding dimension size to reduce to
  --word_select_method WORD_SELECT_METHOD
                        Word selecting methods to select words from each
                        cluster
  --seed SEED           Random seed
```

## Add New Models

To add new topic models, you can inherit the base class `TopicModel` and implement your own `train()`, `evaluate()`, and `get_topics()` functions:

```python
class TopicModel:
    def __init__(self, dataset, topic_model, k):
        self.dataset = dataset
        self.topic_model = topic_model
        self.k = k
        
    def train(self):
        raise NotImplementedError("Train function has not been defined!")

    def evaluate(self):
        raise NotImplementedError("Evaluate function has not been defined!")

    def get_topics(self):
        raise NotImplementedError("Get topics function has not been defined!")
```

## Citation

If our research helps you, please kindly cite our paper:
```
<Placeholder>
```
```
<Placeholder>
```

## Acknowledgement

The code is implemented using [OCTIS](https://github.com/MIND-Lab/OCTIS) and [BERTopic](https://github.com/MaartenGr/BERTopic).

## License

MIT
