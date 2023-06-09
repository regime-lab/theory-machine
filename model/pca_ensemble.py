from .pca import PCA_vectors
import numpy as np 


def fetch_random_assets(factor_base, with_replacement=True):
  ''' fetch a random subset of 30 assets from a pool of factor_base
  '''
  print(factor_base)
  print(len(factor_base))
  r = [int(r) for r in np.random.choice((len(factor_base) - 1), size=(30,), replace=with_replacement)]
  return np.array(factor_base)[r]

def PCA_ensemble(returns, estimators=999):
  ''' finds the heaviest weighted assets in the 1st and 2nd PCA vectors 
      and casts each as a vote over repeated random sampling; TODO
  '''
  votes = [] 
  for n in range(estimators):
      
    subset = fetch_random_assets(returns.columns, with_replacement=False)
    w1, w2 = PCA_vectors(returns[subset])

    votes.append(subset[np.argmax(w1)])
    votes.append(subset[np.argmax(w2)])

  return votes
