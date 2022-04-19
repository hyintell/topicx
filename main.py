import argparse
import logging
logging.getLogger("gensim").setLevel(logging.WARNING)

from baselines.cetopictm import CETopicTM
from baselines.bertopictm import BERTopicTM
from baselines.lda import LDATM
from baselines.prodlda import ProdLDATM
from baselines.zeroshottm import ZeroShotTM
from baselines.combinedtm import CombinedTM
from utils import *

import timeit
import random
import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():

    args = parse_args()
    print(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    dataset, sentences = prepare_dataset(args.dataset)
    print(f'Using dataset: {args.dataset}, number of documents: {len(sentences)}')
    
    
    if args.topic_model == 'cetopic':
        tm = CETopicTM(dataset=dataset, 
                       topic_model=args.topic_model, 
                       k=args.k, 
                       dim_size=args.dim_size, 
                       word_select_method=args.word_select_method,
                       embedding=args.pretrained_model, 
                       seed=args.seed)
    elif args.topic_model == 'lda':
        tm = LDATM(dataset, args.topic_model, args.k)
    elif args.topic_model == 'prodlda':
        tm = ProdLDATM(dataset, args.topic_model, args.k)
    elif args.topic_model == 'combinedtm':
        tm = CombinedTM(dataset, args.topic_model, args.k, args.pretrained_model)
    elif args.topic_model == 'zeroshottm':
        tm = ZeroShotTM(dataset, args.topic_model, args.k, args.pretrained_model)
    elif args.topic_model == 'bertopic':
        tm = BERTopicTM(dataset, args.topic_model, args.k, args.pretrained_model)
    
    tm.train()
    td_score, cv_score, npmi_score = tm.evaluate()
    print(f'Model {args.topic_model} k: {args.k} td: {td_score} npmi: {npmi_score} cv: {cv_score}')
    
    topics = tm.get_topics()
    print(f'Output number of topics: {len(topics.keys())}')
    print(f'Topics: {topics}')
    

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster Contextual Embeddings for Topic Models")
    
    parser.add_argument("--topic_model", type=str, default='lda', help='Topic model to run experiments')
    parser.add_argument("--dataset", type=str, default='20ng', help='Datasets to run experiments')
    parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased', help='Pretrained language model')
    parser.add_argument("--k", type=int, default=10, help='Topic number')
    parser.add_argument("--dim_size", type=int, default=-1, help='Embedding dimension size to reduce to')
    parser.add_argument("--word_select_method", type=str, default='tfidf_idfi', help='Word selecting methods to select words from each cluster')
    parser.add_argument("--seed", type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    return args
    
    
if __name__ == '__main__':
    main()
    