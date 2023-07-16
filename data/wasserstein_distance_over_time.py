import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import gpytorch
import torch 
import seaborn as sns 

from sklearn.cluster import KMeans

import pandas as pd 
import numpy as np 
import scipy.stats as stats
import datetime 
import requests 
from tqdm import tqdm




def get_data(loc_sym, start_date, end_date, ref_index=None): 

  col='close'
  get_year = lambda N: int(N.split('-')[0])
  get_month = lambda N: int(N.split('-')[1])
  get_day = lambda N: int(N.split('-')[2])
  parse_date = lambda D, T: datetime.datetime.combine(datetime.date(
                                get_year(D), 
                                get_month(D), 
                                get_day(D)), 
                              datetime.time(int(T.split(':')[0]), int(T.split(':')[1])))
  parser = lambda row: parse_date(row['date'], row['minute'])

  date_range = pd.date_range(start_date, end_date, freq='d')
  date_range = [ d for d in date_range if d.weekday() not in [5,6] ]
  date_strings = [ str(n).replace('-','').split(' ')[0] for n in date_range ]
  
  OHLC = pd.DataFrame()
  for date in tqdm(date_strings): 
   
    dateCol = 'fullTimestamp'

    try:
      ENDP = f"https://cloud.iexapis.com/stable/stock/{loc_sym}/chart/date/{date}?token=TODO" 
      ohlc_df = None
      ohlc = requests.get(ENDP).json()
      ohlc_df = pd.DataFrame.from_records(ohlc)
   
      #ohlc_df.to_csv(f'data/iex/{loc_sym}_{date}.csv')
      ohlc_df[dateCol] = ohlc_df.apply(parser, axis=1)
      
      if col in ohlc_df \
                  and not ohlc_df[col].isnull().all():
            
        OHLC = OHLC.append(ohlc_df)
        print(OHLC['marketClose'])

    except Exception as e:
      print(e)
  
  OHLC.set_index(OHLC[dateCol], inplace=True)
  OHLC = OHLC[~OHLC.index.duplicated()]

  if ref_index is not None: 
    OHLC = OHLC.reindex(ref_index, method='ffill')

  OHLC[col] = OHLC[col] \
                .replace([np.inf, -np.inf, np.nan, 0.0], method='ffill')

  return OHLC

def get_barset(symbol, symbol_stdate, symbol_eddate, ref_index=None):

  cls = 'close' 
  ohlc_df = get_data(symbol, symbol_stdate, symbol_eddate, ref_index=ref_index)
  local_barset = pd.DataFrame()
  local_barset['Close'] = ohlc_df[cls] 
  local_barset['Volume'] = ohlc_df['marketVolume']
  local_barset['Timestamp'] = ohlc_df['fullTimestamp']
  
  return local_barset, local_barset.index
    
def get_logrets(symbol_base, start_date, end_date, assets=None):

  log_returns = pd.DataFrame()
  base_ohlc = pd.DataFrame()
  _, IDX = get_barset(symbol_base, start_date, end_date)

  for p in assets: 
    try:
      b, _ = get_barset(p, start_date, end_date, ref_index=IDX)
      closes = b['Close']

      base_serie = pd.Series(data=closes, name=p)
      log_serie = pd.Series(data=closes, name=p)

      log_returns = pd.concat([log_returns, log_serie], axis=1, ignore_index=False)
      base_ohlc = pd.concat([base_ohlc, base_serie], axis=1, ignore_index=False)
    except \
      Exception as e: 
        print(e)

  return log_returns, base_ohlc



### PARAMS ### 

SYMBOL = 'IWM'
symbol_stdate = "2023-7-4"
symbol_eddate = "2023-7-14"  

def fetch_assets(assets): 

  log_returns = None
  log_returns, _ = get_logrets(SYMBOL, symbol_stdate, symbol_eddate, assets=assets)
  #aggreg = log_returns.groupby(log_returns.index // GROUP_LEN)
  return log_returns

assetlist = [ 'QQQ', 'NVDA' ]
assets = fetch_assets(assetlist)

# Apply z-score in a rolling way that does not create lookahead bias 
windowed_fn = lambda serie: stats.zscore(serie).values[-1] 
wlen=180

# Clean data
m6_subset1 = assets.replace([np.inf, -np.inf], np.nan).dropna().reset_index().drop(columns='index')
time_series1 = m6_subset1[assetlist[0]] .diff().dropna().values                                                                                   
time_series2 = m6_subset1[assetlist[1]] .diff().dropna().values      
                                              # .rolling(wlen).apply(windowed_fn)         
previous_timeseries = None
previous_eigenvalues = None
wasserstein_distance_over_time = []

for ts_block in range(wlen, len(time_series2), wlen):  

  # Offset TODO     
  local_time_series2 = time_series2[ts_block - wlen: ts_block]     
    
  # Evaluate kernel self similarity matrix 
  kernel = gpytorch.kernels.RBFKernel(lengthscale=10)
  eigenvalues, _ = np.linalg.eig(
    (kernel(torch.tensor(local_time_series2))
                 .evaluate()).detach().numpy() 
    )  

  exp1 = [float(x) for x in eigenvalues]
  if previous_eigenvalues is not None:     
    
    wasserstein_distance_over_time.append(stats.wasserstein_distance(exp1, previous_eigenvalues))
    print(wasserstein_distance_over_time[-1])
    print('wasserstein^')
    sns.lineplot(data=local_time_series2)
    sns.lineplot(data=previous_timeseries, label='prev')
    plt.legend()
    plt.show()
    
  previous_eigenvalues=exp1
  previous_timeseries=local_time_series2
  
sns.lineplot(data=wasserstein_distance_over_time)
plt.show()
