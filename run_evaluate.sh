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
