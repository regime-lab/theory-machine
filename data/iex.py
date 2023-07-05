import pandas as pd 
import numpy as np 
import os 
import sys 
import time 
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
    #dateCol = 'date'
    
    try:
      ENDP = f""
      
      ohlc_df = None
      if os.path.isfile(f'data/iex/{loc_sym}_{date}.csv'):
        ohlc_df = pd.read_csv(f'data/iex/{loc_sym}_{date}.csv')
      else:
        ohlc = requests.get(ENDP).json()
        print(OHLC)
        ohlc_df = pd.DataFrame.from_records(ohlc)
        ohlc_df.to_csv(f'data/iex/{loc_sym}_{date}.csv')

      ohlc_df[dateCol] = ohlc_df.apply(parser, axis=1)
      
      if col in ohlc_df \
                  and not ohlc_df[col].isnull().all():
        OHLC = OHLC.append(ohlc_df)

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

  cls = 'marketClose' 
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
      log_serie = pd.Series(data=np.diff(np.log(closes)), name=p)

      log_returns = pd.concat([log_returns, log_serie], axis=1, ignore_index=False)
      base_ohlc = pd.concat([base_ohlc, base_serie], axis=1, ignore_index=False)
    except \
      Exception as e: 
        print(e)

  return log_returns, base_ohlc



### PARAMS ### 

SYMBOL = 'IWM'
symbol_stdate = "2022-4-10"
symbol_eddate = "2023-5-12" 

DECAPORT = [ 'TLT', 'USO', 'UUP', 'SPY', 'GLD' ]
OUTOF_SAMPLE_STDATE = '2022-4-13'          
OUTOF_SAMPLE_EDDATE = '2022-4-22'            

def fetch_assets(assets, invalidate_cache=False): 

  agg_returns = pd.DataFrame()
  agg_resampler = pd.DataFrame() 
  mean_win = 350
  # 350
  # TODO Gaussian if you start 6-17-2020?
  
  pca_df = None
  base_ohlc = None
  log_returns = None

  if not invalidate_cache and os.path.isfile('data/small_caps/small_caps_resampled.csv'):
    agg_resampler = pd.read_csv('data/small_caps/small_caps_resampled.csv')
    pca_df = agg_resampler.apply(np.log) \
                          .apply(np.diff) \
                          .drop(columns=['date'])  
    pca_df = pca_df.drop(columns=['Unnamed: 0'])
    return (None, None, pca_df)
  
  log_returns, base_ohlc = get_logrets(SYMBOL, symbol_stdate, symbol_eddate, assets=assets)
  base_ohlc = base_ohlc.reset_index().drop(columns='index')

  for col in base_ohlc.columns:
    try:
      base = base_ohlc[col].groupby(base_ohlc.index // mean_win)
      aggreg = log_returns[col].groupby(log_returns.index // mean_win)

      agg_resampler[col] = base.last()#last()
      agg_returns[col] = aggreg.mean()
      pca_df = agg_resampler \
                    .apply(np.log) \
                    .apply(np.diff) \
                    .replace([np.inf,np.nan,-np.inf], 0.0) \
                    .apply(stats.zscore) 
      print(pca_df)

    except Exception as ex:
      print(ex)
  
  return (agg_resampler, agg_returns, pca_df)

