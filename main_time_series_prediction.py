"""Time-series prediction main function

Author: Jinsung Yoon edited to include monte carlo drop out inference and generates the MC estimnate by Hafsa Maryam.

------------------------------------
(1) Load data
(2) Train model (RNN, GRU, LSTM)
(3) Evaluate the trained model
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
import matlab
import matlab.engine

warnings.filterwarnings("ignore")

from data_loader_abilene import data_loader
from basic_rnn_lstm_gru import GeneralRNN
from basic_attention import Attention
from utils import performance


def main (args):  
  """Time-series prediction main function.
  
  Args:
    - train_rate: training data ratio
    - seq_len: sequence length
    - task: classification or regression
    - model_type: rnn, lstm, gru, or attention
    - h_dim: hidden state dimensions
    - n_layer: number of layers
    - batch_size: the number of samples in each mini-batch
    - epoch: the number of iterations
    - learning_rate: learning rates
    - metric_name: mse or mae
  """
  train_x, train_y, test_x, test_y, test_y_fluctuations = data_loader(args.train_rate, 
                                                 args.seq_len, args.source)
  
  #Save text for the true values.
  np.savetxt("ground_truth_0.csv", test_y)
  
  def predict_dist(X, model, num_samples):
        preds = [model(X, training=True) for _ in range(num_samples)]
        return np.hstack(preds)
  def predict_point(X, model, num_samples):
        pred_dist = predict_dist(X, model, num_samples)
        return pred_dist.mean(axis=1)
    
  def predict_class(X, model, num_samples):
        proba_preds = predict_dist(X, model, num_samples)
        return np.argmax(proba_preds, axis=1)
  
    
  # Model traininig / testing
  model_parameters = {'task': args.task,
                      'model_type': args.model_type,
                      'h_dim': args.h_dim,
                      'n_layer': args.n_layer,
                      'batch_size': args.batch_size,
                      'epoch': args.epoch,
                      'learning_rate': args.learning_rate,
                      'loss_function':args.loss_function,
                      'topology':args.topology,
                      'MC_threshold':args.MC_threshold,
                      'source': args.source}
  
  if args.model_type in ['rnn','lstm','gru']:
    general_rnn = GeneralRNN(model_parameters)    
    vv = general_rnn.fit(train_x, train_y)
    test_y_hat = general_rnn.predict(test_x)
  
 
  if args.task == 'regression':
      np.savetxt("MSE_0.csv ", test_y_hat)
      
      
  elif args.task == 'montecarlo_regression':
      MC_estimate_prediction=[]
      print("shape",test_y_hat.shape)
      ytest_value  = test_y.shape[0]
      eng = matlab.engine.start_matlab()
      ytest_value  = test_y_hat.shape[0]
      pred_y_hats=np.array(test_y_hat.reshape(ytest_value,1),).astype('float64')
      
      if args.topology == 'abilene':
         threshold= args.MC_threshold 
         
         y_pred_dist = predict_dist(test_x, vv, 1000)
         y_pred = predict_point(test_x, vv, 1000)
         
         """
         Matlab Script for finding the MC_estimate for the threshold (0.90, 0.95)
         """
         vvs=[]
       
         for i in range(0,len(test_y_hat)):
                         
                         validations_quantile  = eng.MC_estimate(matlab.double(pred_y_hats[i].tolist(), is_complex=True),matlab.double([threshold]))
                         print(validations_quantile,"validations_quantile")
                         vvs.append(validations_quantile)
                         np.savetxt("MC_estimate.csv", vvs)
             
        
       
      elif args.topology == 'geant':
         threshold= args.MC_threshold
         
         y_pred_dist = predict_dist(test_x, vv, 1000)
         y_pred = predict_point(test_x, vv, 1000)
         
         """
         Matlab Script for finding the MC_estimate for the threshold (0.90, 0.95)
         """ 
             
         vvs=[]
       
         for i in range(0,len(test_y_hat)):
                         
                         validations_quantile  = eng.MC_estimate(matlab.double(pred_y_hats[i].tolist(), is_complex=True),matlab.double([threshold]))
                         print(validations_quantile,"validations_quantile")
                         vvs.append(validations_quantile)
                         np.savetxt("MC_estimate.csv", vvs)

                

  
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_rate',
      help='training data ratio',
      default=0.8,
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=6,
      type=int)
  parser.add_argument(
      '--source',
      help='source node',
      default=2,
      type=int)
  parser.add_argument(
      '--model_type',
      choices=['rnn','gru','lstm'],
      default='gru',
      type=str)
  parser.add_argument(
      '--h_dim',
      default=25,
      type=int)
  parser.add_argument(
      '--n_layer',
      default=1,
      type=int)
  parser.add_argument(
      '--batch_size',
      default=50,
      type=int)
  parser.add_argument(
      '--epoch',
      default=500,
      type=int)
  parser.add_argument(
      '--learning_rate',
      default=0.001,
      type=float)
  parser.add_argument(
      '--task',
      choices=['classification','regression','montecarlo_regression'],
      default='regression',
      type=str)
  parser.add_argument(  
      '--loss_function',
      choices=['mse'], # MSE applies for regression, MC applies only for montecarlo_regression,and rnn, gru, lstm models 
      default='mse',
      type=str)
  parser.add_argument(
      '--metric_name',
      choices=['mse','mae'],
      default='mse',
      type=str)
  
  parser.add_argument(
      '--MC_threshold',
      choices=['0.90','0.95'],
      default=0.90,
      type=str)
  
  parser.add_argument(  
      '--topology',
      choices=['abilene','geant'], # active if topology is chosen
      default='geant',
      type=str)
  args = parser.parse_args() 
  
  # Call main function  
  main(args)