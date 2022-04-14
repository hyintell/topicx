# Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics
This is the official repository of NAACL 2022 paper "Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics".
Paper is available at [Placeholder](https://google.com).

## How to Run

```bash
conda create -n cluster_topic_model python=3.7 -y
conda activate cluster_topic_model
pip install -r requirements.txt
bash run_evaluate.sh
```

### Dataset

We use preproessed data from [OCTIS](https://github.com/MIND-Lab/OCTIS#datasets-and-preprocessing). You can choose from `[bbc, m10, 20ng]`. 

### Topic Model
You can run our model `cetopic` or you can also choose a baseline model from `[lda, prodlda, zeroshottm, combinedtm, bertopic]`.

### Word Selecting Method

If you use `cetopic`, you can also choose a word selecting method from `[tfidf_idfi, tfidf_tfi, tfidfi, tfi]`.

### Pretrained Model
You can choose a pretrained model such as `princeton-nlp/unsup-simcse-bert-base-uncased` or `bert-base-uncased` from [SimCSE](https://github.com/princeton-nlp/SimCSE) or [HuggingFace](https://huggingface.co/models).



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