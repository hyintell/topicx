from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import scipy.sparse as sp


class TFIDFi(TfidfTransformer):

    def __init__(self, X_per_cluster, *args, **kwargs):
        print('====== Using TFIDFi ======')
        super().__init__(*args, **kwargs)
        self.X_per_cluster = X_per_cluster
        
    
    def socre(self):
        
        self._tfidfi = self.fit_transform(self.X_per_cluster)
        scores = sp.csr_matrix(self._tfidfi)
        
        return scores 

