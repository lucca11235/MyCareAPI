from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import butter, lfilter

def process(arr):

  ''''
  First step:
    - Split first 15 seconds into 3 second window
    - Find the standard deviation for each window
    - Find the mean of the stds of the windows
    - Calculate threshold
  '''
  big_window = arr[:150]  # Get the first 15 seconds of data
  stds = []
  for small_window in np.array_split(big_window, 5):  # Get the 5 small 3 seconds of data windows
      stds.append(np.std(small_window))

  threshold = 1.2 * np.mean(stds)  # Find threshold

  ''''
  Second step:
    - Create a moving windows
    - As it "grows", calculate the current standard deviation (curr_std) of the values in the window
    - If the curr_std passes the threshold, save the index,then "restart" the window at the next index
  '''

  curr_idx = 0
  cut_threshold_indices = []
  for i in range(1,len(arr)):
      std = np.std(arr[curr_idx:i])
      if std > threshold:
          curr_idx = i
          cut_threshold_indices.append(i)


  ''''
  Third step:
    - Split the array according to the indices found (if there were any)
    - Subtract from each value the mean of the window it was in
    - Concatenate every window
  '''
  # Split the array into windows based on the cut-off indices
  windows = np.split(arr, cut_threshold_indices)

  # Subtract the mean of each window from the values in the window
  for i, window in enumerate(windows):
      windows[i] = window - np.mean(window)

  # Concatenate the windows into a final array of the same length
  final_array = np.concatenate(windows)
  return final_array
from scipy.signal import butter, lfilter


def low_pass_filter(arr, fc, fs):
    # Normalize the cutoff frequency
    w = fc / (fs / 2)

    # Create a first-order Butterworth low pass filter
    b, a = butter(1, w, 'low')

    # Apply the filter to the input signal
    filtered_signal = lfilter(b, a, arr)

    return filtered_signal


def CardioFIPreProcessing(window):


  q = 10
  decimated  = np.apply_along_axis(lambda x: signal.decimate(x, q),
                              axis = 1,
                              arr = window)

  low_pass = np.apply_along_axis(lambda x: low_pass_filter(x, 2, 10),
                             axis = 1,
                             arr = decimated )
  # hampel = np.apply_along_axis(lambda x: hampel_filter(x, window_size=50, n_sigmas=0.4),
  #                            axis = 1,
  #                            arr = low_pass)

  processed = np.apply_along_axis(lambda x: process(x),
                             axis = 1,
                             arr = low_pass)

  return processed


def diffPhaseCorrected(data,
                       fix_val = True,
                       alpha = 0.2,
                       sum = False,
                       conv = None,
                       takes_abs = False):
  '''
  Input: data loaded like np.load(path + f)
  Ouput: data preprocessed for the difference of phases
  '''

  data = data.T
  a1,a2,a3 = data[:57],data[57:114],data[114:]

  # Phase difference
  diff1 = np.angle(np.conj(a1) * a2)
  diff2 = np.angle(np.conj(a2) * a3)
  diff3 = np.angle(np.conj(a1) * a3)

  # Unwrap it
  diff1 = np.unwrap(diff1,axis = 0)
  diff2 = np.unwrap(diff2,axis = 0)
  diff3 = np.unwrap(diff3,axis = 0)

  # Scale
  scaler = MinMaxScaler()

  diff1 = scaler.fit_transform(diff1)
  diff2 = scaler.fit_transform(diff2)
  diff3 = scaler.fit_transform(diff3)

  # Fix the value problem
  if fix_val:
    # Fix the value problem
    def normalize(x): return (1 - (np.exp(-(x-0.5)**2)/0.2)/2)
    diff1 = normalize(diff1)
    diff2 = normalize(diff2)
    diff3 = normalize(diff3)

  # subtract the mean
  diff1 = diff1 - np.mean(diff1, axis=0)
  diff2 = diff2 - np.mean(diff2, axis=0)
  diff3 = diff3 - np.mean(diff3, axis=0)

  if conv is not None:
    kernel = [-1]*conv + [1]*conv
    diff1 = applyConv(diff1,kernel)
    diff2 = applyConv(diff2,kernel)
    diff3 = applyConv(diff3,kernel)

  # Concatenate and return
  if takes_abs:
    diff1 = abs(diff1)
    diff2 = abs(diff2)
    diff3 = abs(diff3)

  if sum:
    final = diff1 + diff2 + diff3
  else:
    final = np.concatenate((diff1,diff2,diff3), axis = 0)

  return final
