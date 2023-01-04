# Uncertainty Quantification and Consideration in ML-aided Traffic-Driven Service Provisioning

In this work, we examine traffic prediction uncertainty  by leverging the capabilites of Monte Carlo (MC) dropout inference for reoptimizing the network resources, provisioinig of services with diverse QoS requirements in optical networks. Also, we have investiagted predictive distribution of MC dropout inference to provide prediction under various certainty levels. We adopt a deep MC regression framework. In this framework, Deep Neural Networks (DNNs) are trained to estimate MC dropout inference function (i.e., models). Further, simulations are performed on real-world traffic traces from the Abilene and Geant network for calculating the spectrum savings. 


## Python Script
This directory contains implementations of basic traffic prediction using RNN, GRU, LSTM methods. To run the pipeline, simply run python3 -m main_time_series_prediction.py.

  ## DNN Code Description
  It contains eight files, which are:

  * **data_loader.py** This file loads the Abilene/Geant bit-rate dataset and performs MinMax normalization.

  * **data_laoder_abilene.py** This file generates the 800 for Abilene and 133 for Geant sequential (in time) traffic patterns and performs network reconfiguration k=6 fluctuations within each planning interval considering the input patterns.

  * **basic_rnn_lstm_gru.py** This files contains the MC dropout inference fucntion and prediction model to generates the MSE loss fucntion.

  * **main_time_series_prediction.py** This file generates the predictions for MSE and MC estimates for Monte Carlo dropout inference, 0.90 and 0.95 certainty           thresholds.  

  ## DNN Dataset

  The dataset folder contains two subfolders are:

  ### Abilene  
  This folder contains four files are:
  * **source_data_abilene** This csv file contains the bit-rates in Gbps for all nodes.
  * **source** This csv file contains all the source nodes.

  ### Geant 
  This folder contains four files are:
  * **source_data_geant** This csv file contains the bit-rates in Gbps for all nodes.
  * **source** This csv file contains all the source nodes.

## Code Example

## Steps of Traffic Prediction framework:
**Step 1:** Load bit-rate dataset for Abilene/Geant network by simply running the *data_laoder.py*.
     * **Abilene:** To load data (source_data_abilene), put range (0,4000) sequential (in time) traffic patterns.
     * **Geant:** To load data (source_data_geant), put range (0,2000) sequential (in time) traffic patterns.0, 2000  sequential (in time) traffic patterns.
     
**Step 2:** Load bit-rate dataset per source node for Abilene/Geant network for testing and training by simply running the *data_laoder_abilene.py*.
     * Abilene: 0,4000 sequential (in time) traffic patterns.
     * Geant: 0, 2000  sequential (in time) traffic patterns.

## Referecence

### We kindly ask that if you use our dataset or code,  please reference the following paper: 
[1]  H. Maryam,T. Panayiotou, and G. Ellinas, "Uncertainty Quantification and Consideration in ML-aided Traffic-Driven Service Provisioning", Computer Communications, 2022.

## Accknowledgment
This work has been supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 739551 (KIOS CoE) and from the Government
of the Republic of Cyprus through the Directorate General for European Programmes, Coordination and Development.
