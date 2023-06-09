import numpy as np
import statsmodels.api as sm

def get_linear_tval(series):
    ''' Fit an ordinary least squares regression to series and return t-statistic. 
        Returns: 
            t-values 
            params 
    '''
    x=np.ones((series.shape[0],2))
    x[:,1]=np.arange(series.shape[0])
    
    ols=sm.OLS(series, x).fit()
    return  ols.tvalues[1], \
            ols.params[1]
