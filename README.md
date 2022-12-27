# Uncertainty Quantification and Consideration in ML-aided Traffic-Driven Service Provisioning

In this work, we examine traffic prediction uncertainty  by leverging the capabilites of Monte Carlo (MC) dropout inference for reoptimizing the network resources, provisioinig of services with diverse QoS requirements in optical networks. Also, we have investiagted predictive distribution of MC dropout inference to provide prediction under various certainty levels. We adopt a deep MC regression framework. In this framework, Deep Neural Networks (DNNs) are trained to estimate MC dropout inference function (i.e., models). Further, simulations are performed on real-world traffic traces from the Abilene and Geant network for calculating the spectrum savings. 


## Python Script

## DNN Code Description
It contains seven files, which are:

* **load_txt.py** This file contains the Python script and the code creates the abilene/geant csv file (i.e., bit-rates along with the source nodes).

* **create_source_data.py** This file load the bit-rates and generates the bit-rates in Gbps of all source nodes for abilene/geant.

* **data_loader.py** This file loads the Abilene/Geant bit-rate dataset and performs MinMax normalization.

* **data_laoder_abilene.py** This file genrates the 800 for Abilene and 133 for Geant sequential (in time) traffic patterns and performs network reconfiguration k=6 fluctuations within each planning interval considering the input patterns.

* **basic_rnn_lst_gru.py** This files conatins the MC dropout inference fucntion and prediction model to generates the MSE loss fucntion.

* **main_time_series_prediction.py** This file generates the estimates (i.e., prediction and MC estimates) for 0.90 and 0.95 certainty thresholds.  

* **basic_attenstion.py** This file coantians are core functions.

## DNN Dataset

The code folder conatinas two files are:
* **qunatile_regression.py** This file contains the Python script and the code generates loss functions in deep quantile neural network to evaluate the error estimation and performance accuracy for 0.1 and 0.95 quantile QoT models.

*  **least_squares_regression.py**  This file generates the MSE loss function to generate the estimate (i.e., prediction). 

Each line is a vector v describing the features and the QoT value of a lightpath.  Specifically:
v = [v_1, v_2, v_3, v_4, v_5, v_6, v_7],
where
* v_1: is the path length of the lightpath
* v_2: is the number of hops (or links) of the lightpath
* v_3: is the degree of the destination node of the lightpath 
* v_4: is the maximum link length of the lightpath
* v_5: is the wavelength allocated to the lightpath
* v_6: is Q-factor of the lightpath

Hence, v_1  to v_5 are the inputs of the DNN model, and v_6 is the output of the model. 


## Referecence

### We kindly ask that if you use our dataset or code,  please reference the following paper: 
[1]  H. Maryam,T. Panayiotou, and G. Ellinas, "Uncertainty Quantification and Consideration in ML-aided Traffic-Driven Service Provisioning", Computer Communications, 2022.

## Accknowledgment
This work has been supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 739551 (KIOS CoE) and from the Government
of the Republic of Cyprus through the Directorate General for European Programmes, Coordination and Development.
