import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA


# https://stats.stackexchange.com/questions/247260/principal-component-analysis-eliminate-noise-in-the-data

get_partition = lambda vec, N: np.argpartition(vec, -N)[-N:]

reduce_topN = lambda vec, part: [ 1/len(part) if M in part else 0. for M in range(len(vec)) ]

reduce_eigen = lambda vec: [ 1. if np.argmax(vec) == M else 0. for M in range(len(vec)) ]

def PCA_vectors(rets, num_components=4, reduce=False):
  cols = rets.columns.unique()  
  feat_num = len(cols)
  eigenvecs = [] 

  pca = PCA(n_components=num_components)  
  pca.fit_transform(rets[cols].values.reshape((len(rets), feat_num)))

  for M in range(num_components):
    eigenvec = pd.Series(index=range(feat_num), data=pca.components_[M])
    eigenvec = abs(eigenvec) / sum(abs(eigenvec))
    
    prt = get_partition(eigenvec, 1).values

    eigenvecs.append(
      reduce_topN(eigenvec, prt)
        if reduce 
        else eigenvec)

  return eigenvecs[0], eigenvecs[1], eigenvecs[2], eigenvecs[3]
