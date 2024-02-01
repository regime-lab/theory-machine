import numpy as np

def bartletts_method(data, segment_len):
    ''' https://en.wikipedia.org/wiki/Bartlett%27s_method
    A method of averaging periodograms across the data.
    '''
    # data length and number of segments
    k = len(data) // segment_len 
  
    # reshape into segments of length segment_len
    segments = np.reshape(data[:k * segment_len], (k, segment_len))
  
    # periodogram for each segment then averaged 
    periodograms = np.abs(np.fft.fft(segments, axis=1)) ** 2 / segment_len
    return np.mean(periodograms, axis=0)


def frequency_diagram(signal, segment_len=64):
  
  # get result
  result = bartletts_method(signal, segment_len)
  
  # return result with bins
  freq_bins = np.fft.fftfreq(segment_len)[:segment_len//2]
  return freq_bins, result
