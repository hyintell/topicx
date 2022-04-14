DATASET='bbc'
# DATASET='m10'
# DATASET='20ng'

# TOPIC_MODEL='lda'
# TOPIC_MODEL='prodlda'
# TOPIC_MODEL='zeroshottm'
# TOPIC_MODEL='combinedtm'
TOPIC_MODEL='cetopic'
# TOPIC_MODEL='bertopic'

WORD_SELECT_METHOD='tfidf_idfi'
# WORD_SELECT_METHOD='tfidf_tfi'
# WORD_SELECT_METHOD='tfidfi'
# WORD_SELECT_METHOD='tfi'

# PRETRAINED_MODEL='sentence-transformers/stsb-bert-base'
PRETRAINED_MODEL='bert-base-uncased'
# PRETRAINED_MODEL='princeton-nlp/unsup-simcse-bert-base-uncased'
# PRETRAINED_MODEL='princeton-nlp/sup-simcse-roberta-base'


python main.py\
    --topic_model ${TOPIC_MODEL}\
    --dataset ${DATASET}\
    --k 5\
    --dim_size 5\
    --word_select_method ${WORD_SELECT_METHOD}\
    --pretrained_model ${PRETRAINED_MODEL}\
    --seed 30