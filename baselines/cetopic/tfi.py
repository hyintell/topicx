from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp


class TFi(TfidfTransformer):

    def __init__(self, X_per_cluster, *args, **kwargs):
        print('====== Using TFi ======')
        super().__init__(*args, **kwargs)
        self.X_per_cluster = X_per_cluster
        
    
    def socre(self):
        
        self._tfi = normalize(self.X_per_cluster, axis=1, norm='l1', copy=False)
        scores = sp.csr_matrix(self._tfi)
        
        return scores 

