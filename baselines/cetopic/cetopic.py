"""
Implemented based on BERTopic https://github.com/MaartenGr/BERTopic
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csr import csr_matrix

from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from .tfi import TFi
from .tfidfi import TFIDFi
from .tfidf_idfi import TFIDF_IDFi
from .tfidf_tfi import TFIDF_TFi
from .backend._utils import select_backend


class CETopic:

    def __init__(self, top_n_words=10, nr_topics=10, embedding_model=None, dim_size=-1, word_select_method=None, seed=42):
        
        self.topics = None
        self.topic_sizes = None
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.word_select_method = word_select_method
        self.embedding_model = embedding_model
        self.vectorizer_model = CountVectorizer()
        
        self.dim_size = dim_size
        self.umap = None
        if self.dim_size != -1:
            self.umap = UMAP(n_neighbors=15, n_components=self.dim_size, min_dist=0.0, metric='cosine')
        
        # cluster
        self.kmeans = KMeans(self.nr_topics, random_state=seed)

        
    def fit_transform(self, documents, embeddings=None):
        
        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        if embeddings is None:
            self.embedding_model = select_backend(self.embedding_model)
            embeddings = self._extract_embeddings(documents.Document)
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(self.embedding_model)

        if self.umap is not None:
            embeddings = self._reduce_dimensionality(embeddings)
        
        documents = self._cluster_embeddings(embeddings, documents)

        self._extract_topics(documents)
        predictions = documents.Topic.to_list()

        return predictions


    def get_topics(self):
        return self.topics
    

    def get_topic(self, topic_id):
        if topic_id in self.topics:
            return self.topics[topic_id]
        else:
            return False


    def _extract_embeddings(self, documents):
        
        embeddings = self.embedding_model.embed_documents(documents)

        return embeddings
    

    def _reduce_dimensionality(self, embeddings):

        self.umap.fit(embeddings)
        reduced_embeddings = self.umap.transform(embeddings)
        
        return np.nan_to_num(reduced_embeddings)
    

    def _cluster_embeddings(self, embeddings, documents):
        
        self.kmeans.fit(embeddings)
        documents['Topic'] = self.kmeans.labels_
        self._update_topic_size(documents)

        return documents


    def _extract_topics(self, documents):
        
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        self.scores, words = self._weighting_words(documents_per_topic, documents)
        self.topics = self._extract_words_per_topic(words)


    def _weighting_words(self, documents_per_topic, all_documents):
        
        concatenated_documents = self._preprocess_text(documents_per_topic.Document.values)
        origin_documents = self._preprocess_text(all_documents.Document.values)
        
        # count the words in a cluster
        self.vectorizer_model.fit(concatenated_documents)
        words = self.vectorizer_model.get_feature_names()
        
        # k * vocab
        X_per_cluster = self.vectorizer_model.transform(concatenated_documents)
        # D * vocab
        X_origin = self.vectorizer_model.transform(origin_documents)
        
        if self.word_select_method == 'tfidf_idfi':
            socres = TFIDF_IDFi(X_per_cluster, X_origin, all_documents).socre()
        elif self.word_select_method == 'tfidf_tfi':
            socres = TFIDF_TFi(X_per_cluster, X_origin, all_documents).socre()
        elif self.word_select_method == 'tfi':
            socres = TFi(X_per_cluster).socre()
        elif self.word_select_method == 'tfidfi':
            socres = TFIDFi(X_per_cluster).socre()

        return socres, words
    

    def _update_topic_size(self, documents):

        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes = dict(zip(sizes.Topic, sizes.Document))
        

    def _extract_words_per_topic(self, words):

        labels = sorted(list(self.topic_sizes.keys()))

        # Get the top 30 indices and values per row in a sparse c-TF-IDF matrix
        indices = self._top_n_idx_sparse(self.scores, 30)
        scores = self._top_n_values_sparse(self.scores, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {label: [(words[word_index], score)
                          if word_index and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}

        topics = {label: values[:self.top_n_words] for label, values in topics.items()}

        return topics


    def _preprocess_text(self, documents):
        """ Basic preprocessing of text

        Steps:
            * Lower text
            * Replace \n and \t with whitespace
            * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.lower() for doc in documents]
        cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
        cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]

        return cleaned_documents
    

    @staticmethod
    def _top_n_idx_sparse(matrix, n):
        """ Return indices of top n values in each row of a sparse matrix

        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

        Args:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row

        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
            indices.append(values)
        return np.array(indices)
    

    @staticmethod
    def _top_n_values_sparse(matrix, indices):
        """ Return the top n values for each row in a sparse matrix

        Args:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row

        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        return np.array(top_values)


