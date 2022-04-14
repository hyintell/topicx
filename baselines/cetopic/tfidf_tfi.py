from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sp


class TFIDF_TFi(TfidfTransformer):

    def __init__(self, X_per_cluster, X_origin, all_documents, *args, **kwargs):
        print('====== Using TFIDF_TFi ======')
        super().__init__(*args, **kwargs)
        self.X_per_cluster = X_per_cluster
        self.X_origin = X_origin
        self.all_documents = all_documents
        
    
    def socre(self):
        
        self._global_tfidf = self.fit_transform(self.X_origin)
        
        global_df = pd.DataFrame(self._global_tfidf.toarray())
        global_df['Topic'] = self.all_documents.Topic
        
        avg_global_df = global_df.groupby(['Topic'], as_index=False).mean()
        avg_global_df = avg_global_df.drop('Topic', 1)
        self._avg_global_tfidf = avg_global_df.values
        
        # k * vocab
        self._tfi = normalize(self.X_per_cluster, axis=1, norm='l1', copy=False).toarray()

        scores = self._avg_global_tfidf * self._tfi
        scores = normalize(scores, axis=1, norm='l1', copy=False)
        scores = sp.csr_matrix(scores)
        
        return scores 

