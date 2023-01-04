"""Data loader.

Author: Jinsung Yoon 
Edited by Hafsa Maryam to include Geant and Abilene bit-rate data.

----------------------------------------
Loads Abilene bit-rate dataset with MinMax normalization.
"""

# Necessary Packages
import numpy as np
from utils import MinMaxScaler


def data_loader(train_rate, seq_len, source):
  """Loads bit-rate data per source node
  
  Args:
    - train_rate: the ratio between training and testing sets
    - seq_len: sequence length
    
  Returns:
    - MC_eatimnate: training feature
    - train_y: training labels
    - test_x: testing features
    - test_y: testing labels
  """
  
  # Load data
  ori_data = np.loadtxt('source_data_abilene.csv', delimiter=',', skiprows = 1) #('source_data_geant.csv') for Geant
  # Reverse the time order
  #reverse_data = ori_data[::-1]
  source_data=ori_data[range(0,4000),source] # (0,2000) for Geant
  # Normalization
  norm_data = MinMaxScaler(source_data)
    
  # Build dataset
  data_x = []
  data_y = []
  
  for i in range(0, len(norm_data) - seq_len):
    # Previous seq_len data as features
    temp_x = norm_data[i:i + seq_len]
    # Values at next time point as labels
    temp_y = source_data[i + seq_len] # not normalized output
    data_x = data_x + [temp_x]
    data_y = data_y + [temp_y]
    
  data_x = np.asarray(data_x)
  data_y = np.asarray(data_y)
            
  # Train / test Division   
  #idx = np.random.permutation(len(data_x))
  idx  = list(range(1, len(data_x))) # sequential idx
  train_idx = idx[:int(train_rate * len(data_x))]
  test_idx = idx[int(train_rate * len(data_x)):]
        
  train_x, test_x = data_x[train_idx, :], data_x[test_idx, :]
  train_x=np.asarray(train_x).reshape(len(train_x),1,seq_len)
  test_x=np.asarray(test_x).reshape(len(test_x),1,seq_len)
  train_y, test_y = data_y[train_idx], data_y[test_idx]
  train_y=np.asarray(train_y).reshape(len(train_y),1)
  test_y=np.asarray(test_y).reshape(len(test_y),1)
    
  return train_x, train_y, test_x, test_y
