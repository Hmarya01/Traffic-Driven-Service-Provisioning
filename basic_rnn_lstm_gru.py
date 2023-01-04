"""General RNN core functions for time-series prediction.

Author: Jinsung Yoon 
Edited by Hafsa Maryam to include Monte Carlo drop out inference

"""

# Necessary packages
import os
import time
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from utils import binary_cross_entropy_loss, mse_loss, rnn_sequential
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt 


class GeneralRNN():
  """RNN predictive model to time-series.
  
  Attributes:
    - model_parameters:
      - task: classification or regression
      - model_type: 'rnn', 'lstm', or 'gru'
      - h_dim: hidden dimensions
      - n_layer: the number of layers
      - batch_size: the number of samples in each batch
      - epoch: the number of iteration epochs
      - learning_rate: the learning rate of model training
  """

  def __init__(self, model_parameters):

    self.task = model_parameters['task']
    self.model_type = model_parameters['model_type']
    self.h_dim = model_parameters['h_dim']
    self.n_layer = model_parameters['n_layer']
    self.batch_size = model_parameters['batch_size']
    self.epoch = model_parameters['epoch']
    self.learning_rate = model_parameters['learning_rate']
    self.loss_function = model_parameters['loss_function']
    self.topology = model_parameters['topology']
    self.source = model_parameters['source']
    
    assert self.model_type in ['rnn', 'lstm', 'gru']

    # Predictor model define
    self.predictor_model = None

    # Set path for model saving
    model_path = 'tmp'
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    self.save_file_name = '{}'.format(model_path) + \
                          datetime.now().strftime('%H%M%S') + '.hdf5'
  

  def _build_model(self, x, y):
    """Construct the model using feature and label statistics.
    
    Args:
      - x: features
      - y: labels
      
    Returns:
      - model: predictor model
    """    
    # Parameters
    h_dim = self.h_dim
    n_layer = self.n_layer
    dim = len(x[0, 0, :])
    max_seq_len = len(x[0, :, 0])

    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=(max_seq_len, dim)))
    
    # Monte-Carlo Drpopout inference
    class MonteCarloDropout(Dropout):
      def call(self, Dropout):
        return super().call(Dropout, training=True)
  
   
    
    
    for _ in range(n_layer - 1):
      model = rnn_sequential(model, self.model_type, h_dim, return_seq=True)

    model = rnn_sequential(model, self.model_type, h_dim,  return_seq=False)
    adam = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.9, amsgrad=True)
   

    if self.task == 'classification':
      model.add(layers.Dense(y.shape[-1], activation='sigmoid'))
      model.compile(loss=binary_cross_entropy_loss, optimizer=adam)
      
      
    elif self.task == 'montecarlo_regression':
        
        
       if self.topology == 'abilene':
            model.add(Dense(units=32))
            model.add(Dense(units=32))
            model.add(MonteCarloDropout(0.1))
            model.add(Dense(y.shape[-1])) 
            model.compile(loss="mse", optimizer=adam, metrics=['mse'])
            
            
       elif self.topology == 'geant':
            
           
           model.add(Dense(units=32))
           model.add(MonteCarloDropout(0.1))
       
           model.add(Dense(y.shape[-1],activation='linear')) 
           model.compile(loss="mse", optimizer=adam, metrics=['mse'])
       
      
    elif self.task == 'regression':
        model.add(layers.Dense(y.shape[-1], activation='relu'))

        if self.loss_function == 'mse':
            model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=adam, metrics=['mse'])
            
       
    return model
   
 
  
    
  def fit(self, x, y):
    """Fit the predictor model.
    
    Args:
      - x: training features
      - y: training labels
      
    Returns:
      - self.predictor_model: trained predictor model
    """
    #idx = np.random.permutation(len(x))
    idx  = list(range(1, len(x)))
    #train_idx = idx[:int(len(idx)*0.8)]
    train_idx = idx[:int(0.8 * len(x))]
    
    #valid_idx = idx[int(len(idx)*0.8):]
    valid_idx = idx[int(0.8 * len(x)):]
    
    train_x, train_y = x[train_idx], y[train_idx]
    valid_x, valid_y = x[valid_idx], y[valid_idx]
    
    self.predictor_model = self._build_model(train_x, train_y)

    # Callback for the best model saving
    save_best = ModelCheckpoint(self.save_file_name, monitor='val_loss',
                                mode='min', verbose=False,
                                save_best_only=True)

    time_start = time.time()
    # Train the model
    history=self.predictor_model.fit(train_x, train_y, 
                             batch_size=self.batch_size, epochs=self.epoch, 
                             validation_data=(valid_x, valid_y), 
                             callbacks=[save_best], verbose=True)
    
    time_elapsed = (time.time() - time_start)
    print (time_elapsed)

    self.predictor_model.load_weights(self.save_file_name)
    os.remove(self.save_file_name)
    
    plt.plot(history.history['loss'],linestyle='dashed')
    plt.plot(history.history['val_loss'])
    plt.ylabel('$L_{dropout}$ for v=1', fontsize='large') # write number according to the source nodes.
    plt.xlabel('Epochs',fontsize='large')
    plt.legend( ['Train', 'Validation'], loc='upper right',fontsize='large')
    plt.show()  
    return self.predictor_model
  
  
    
  def predict(self, test_x):
      
    """Return the temporal and feature importance.
    
    Args:
      - test_x: testing features
      
    Returns:
      - test_y_hat: predictions on testing set
    """
   
    test_y_hat = self.predictor_model.predict(test_x)
    
    return test_y_hat
  