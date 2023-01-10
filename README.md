# Uncertainty Quantification and Consideration in ML-aided Traffic-Driven Service Provisioning

In this work, we examine traffic prediction uncertainty  by leverging the capabilites of Monte Carlo (MC) dropout inference for reoptimizing the network resources, provisioinig of services with diverse QoS requirements in optical networks. Also, we have investiagted predictive distribution of MC dropout inference to provide prediction under various certainty levels. We adopt a traffic predicition framework. In this framework, Deep Neural Networks (DNNs) are trained to estimate MC dropout inference function (i.e., models). Further, simulations are performed on real-world traffic traces from the Abilene and Geant network for calculating the spectrum savings. 


## Python Script
This directory contains implementations of basic traffic prediction using RNN, GRU, LSTM methods. To run the pipeline, simply run python3 -m main_time_series_prediction.py.

  ## DNN Code Description
  It contains five files, which are:

  * **data_laoder_abilene.py** This file generates the 800 for Abilene and 133 for Geant sequential (in time) traffic patterns and performs network reconfiguration k=6 fluctuations within each planning interval considering the input patterns.

  * **basic_rnn_lstm_gru.py** This files contains the MC dropout inference fucntion and prediction model to generates the MSE loss fucntion.

  * **main_time_series_prediction.py** This file generates the predictions for MSE and MC estimates for Monte Carlo dropout inference, 0.90 and 0.95 certainty           thresholds.  
   
   * **MC_estimate.m** This matlab file contains the fucntion to generate the MC estimates from the test patterns for 0.90 and 0.95 certainty thresholds.
   * **utils.py** This file contains all the core function.
   
  ## DNN Dataset

  The dataset folder contains two subfolders are:

  ### Abilene  
  This folder contains two files are:
  * **source_data_abilene** This csv file contains the bit-rates in Gbps for all nodes.
  * **source** This csv file contains all the source nodes.

  ### Geant 
  This folder contains two files are:
  * **source_data_geant** This csv file contains the bit-rates in Gbps for all nodes.
  * **source** This csv file contains all the source nodes.

## Code Example

## Steps of Traffic Prediction framework:

**Step 1:** Load bit-rate dataset per source node for testing and training for Abilene/Geant network by simply running the data_laoder_abilene.py .
* **Abilene:** To load data (source_data_abilene), put range (0,6*800) sequential (in time) traffic patterns.
* **Geant:** To load data (source_data_geant), put range (0,6*133) sequential (in time) traffic patterns.
     
**Step 2:** After laoding the data, to train the model (RNN based: Simple RNN, GRU, LSTM) for the performance evaluation (MSE) in main_time_series_prediction.py file , following commands are important:
- train_rate: training data ratio
- seq_len: sequence length
- task: classification or regression or montecarlo_regression(MC_estimate)
- model_type: rnn, lstm or gru
- h_dim: hidden state dimensions
- n_layer: number of layers
- batch_size: the number of samples in each mini-batch
- epoch: the number of iterations
- source: the number of source node for Abilene/ Geant -  as per demand of the network.
- learning_rate: learning rates
- topology: Abilene or Geant
- threshold: Certainty threshold q value for (0.90 or 0.95) to find MC_estimate.
      
  Afer cheking the above input commands, simply run the main file named "main_time_series_prediction.py" and see the (prediction/MC_estimate) with the loss fucntion for all the source nodes.    
      
## Referecence

### We kindly ask that if you use our dataset or code,  please reference the following paper: 
[1]  H. Maryam,T. Panayiotou, and G. Ellinas, "Uncertainty Quantification and Consideration in ML-aided Traffic-Driven Service Provisioning", Computer Communications, 2022.

## Accknowledgment
This work has been supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 739551 (KIOS CoE) and from the Government
of the Republic of Cyprus through the Directorate General for European Programmes, Coordination and Development.
