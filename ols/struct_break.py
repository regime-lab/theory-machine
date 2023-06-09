import numpy as np 
import pandas as pd 
from .least_squares import get_linear_tval

def get_structBreakCandidates(molecule, close, span): 
  ''' get_structBreakCandidates
        
      these structural breaks are more likely to occur in the variance 
      rather than the mean (large increases in the t-statistic denom-
      inator) and represent jumps in `uncertainty`

      Parameters: 
        molecule      starting timestamp
        close         close indices
        span          look ahead period length 

      Returns:
        dataframe with t-statistics  
  '''
  hrzns=range(span)
  indic_upper = [] 
  indic_lower = [] 

  for dt0 in [0]:

    df0=pd.Series(dtype=float)
    iloc0=close.index.get_loc(dt0)
    if iloc0+max(hrzns)>close.shape[0]:continue

    # Gather up the t-statistic for multiple regression horizons 
    for hrzn in hrzns:
      dt1=close.index[iloc0+hrzn-1]
      df1=close.loc[dt0:dt1]
      if len(df1) > 1: 
        tval,slope=get_linear_tval(df1.values)
        df0.loc[dt1]=tval
       
    return df0 
