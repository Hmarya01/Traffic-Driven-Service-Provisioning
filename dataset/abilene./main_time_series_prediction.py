"""Time-series prediction main function

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

edited by Tania Panayiotou
------------------------------------
(1) Load data
(2) Train model (RNN, GRU, LSTM, Attention)
(3) Evaluate the trained model
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings
import numpy as np
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
  
  #Save text for the true values and true_fluctutions
  #np.savetxt("true_11.csv", test_y/1e18)
  # np.savetxt("MSSEE\ground_truth_222_2.csv", test_y)
  # np.savetxt("fluctuations_011.csv", test_y_fluctuations)
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
                      'loss_function':args.loss_function}
  
  if args.model_type in ['rnn','lstm','gru']:
    general_rnn = GeneralRNN(model_parameters)    
    vv = general_rnn.fit(train_x, train_y)
    test_y_hat = general_rnn.predict(test_x)
    # np.savetxt("quantile_9_0.csv", test_y_hat/1e18)
    # np.savetxt("qqmseMC.csv", test_y_hat)
    
  
  # np.savetxt("MSSEE\MSE_222_2.csv ", test_y_hat)
  # np.savetxt("Qunatile_0.95\Qunatile_11.csv ", test_y_hat)
  
  MC_estimate_prediction=[]
  print("shape",test_y_hat.shape)
  
    # 0-2 node k lye -0.1 krna hai, 3-0.2,  4,5 model k kuch nhn krna, 6,7 men -0.5 krna hai, 8,9,10 men kuch nhn krna
  # with open("Qos_222\MCs" + ".csv", "w") as out_file:
  #                 for i in range(len(test_y_hat)):
  
  #                     #print('arrrayayayya',mae);
  #                     out_string = ""
  #                     #out_string += str(X_test[i])
  #                     out_string += "," + str(test_y_hat[i]) #for prediction
  #                     # out_string += "," + str(test_y_hat[i])
  #                     out_string += "\n"+ str(test_y[i])
    
    
  #                     out_file.write(out_string) 
  
                # general_rnn.predicts(test_x)
                # Evaluation
  import seaborn as sns
  import pandas as pd
  import matplotlib.pyplot as plt 
  result = performance(test_y, test_y_hat, args.metric_name)
  print('Performance (' + args.metric_name + '): ' + str(result))
                # print(test_y_hat, "Predicitions")
  ytest_value  = test_y.shape[0]

                #############################################################################
                # 
                ##############################################################################
                #  # np.savetxt("true_value.csv", test_y)
                #  # models = GeneralRNN.montecarlomode(test_x)
                # #  v = general_rnn._build_model(train_x, train_y)
  y_pred_dist = predict_dist(test_x, vv, 1000)
  y_pred = predict_point(test_x, vv, 1000)
  sns.kdeplot(y_pred_dist[0], shade=True)
  plt.axvline(y_pred[0], color='red')
  plt.axvline(test_y[0], color='green')
               
  plt.ylabel('Predictive Distribution for $x_{*}$ ',fontsize='large')
  plt.show()
  
   
#   with open("Qos_222\MCprediction" + ".csv", "w") as out_file:
#                       for i in range(len(y_pred)):

#                           #print('arrrayayayya',mae);
#                           out_string = ""
#                           #out_string += str(X_test[i])
#                           out_string += "," +  str(y_pred[i]) #for prediction
# #out_string += "," + str(y_pred[i]-0.2,3-0.2,5-0,6-0.5,8-0,9-0.05, 10-0, 11-0.02),
#                           out_string += "," + str(y_pred[i]-0.2)
#                           out_string += "," + str(test_y_hat[i])
#                           out_string += "," + str(test_y_hat[i]-0.2)
#                           #out_string += "," + str(test_y_hat[i]-0.5)
#                           out_string += "\n"+ str(test_y[i])
      
      
#                           out_file.write(out_string)   
   
  import matlab
  import matlab.engine
  eng = matlab.engine.start_matlab()
  ytest_value  = test_y_hat.shape[0]
  pred_y_hats=np.array(test_y_hat.reshape(ytest_value,1),).astype('float64')
  threshold= 0.90
  vvs=[]
  kvs=[]
          # k=5
          # print("len(pred_y_hat)",k)
                  # print(pred_y_hat) 
  for i in range(0,len(test_y_hat)):
                    for x in range(i+1):
                        print(test_y_hat[x], x , pred_y_hats[x])
                    #kvs.append(pred_y_hats[x]-0.5)
                    kvs.append(pred_y_hats[x])
                    print(kvs)
                    # np.savetxt("Qos_222\MC.csv", kvs)
                    # np.savetxt("MC_prediction_dropout_3_0.95layer_21.csv", kvs)
                    # ytest_value  = kvs.shape[0]
                    # pred_y_hat=np.array(kvs.reshape(ytest_value,1),).astype('float64')
                    validations_quantile  = eng.MC_estimate(matlab.double(kvs[i].tolist(), is_complex=True))
                    vvs.append(validations_quantile)
                    print(validations_quantile,"validations_quantile")
                    vvs.append(validations_quantile)
                    # np.savetxt("Qos_222\MC_estimates_0.90.csv", vvs)
  # np.savetxt("true_values.csv", test_y)  

  
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
      default=0,
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
      default='montecarlo_regression',
      type=str)
  parser.add_argument(  
      '--loss_function',
      choices=['mse'], # quantile applies only for regression and rnn, gru, lstm models 
      default='mse',
      type=str)
  parser.add_argument(
      '--metric_name',
      choices=['mse','mae'],
      default='mse',
      type=str)
  
  args = parser.parse_args() 
  
  # Call main function  
  main(args)