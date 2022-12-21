# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:58:18 2021

@author: eep5pt1

creates the abilene csv file
"""
import os
import pandas as pd  
import numpy as np



path = 'data/abilene'

nodes_all=[ 'ATLAM5', 'ATLAng' , 'CHINng', 'DNVRng' , 'HSTNng', 'IPLSng', 'KSCYng', 'LOSAng' , 'NYCMng', 'SNVAng', 'STTLng', 'WASHng'] 
nodes=np.array(nodes_all)
data_txt=os.listdir(path)

cols = ["source", "bit_rate"]
rows = []



for filename in data_txt:
     fullname = os.path.join(path, filename)
     with open(fullname) as f:
         lines = f.readlines()
         rates = np.zeros(len(nodes))
         for i in range(45,len(lines)):
             arr=lines[i].split()
             if len(arr)==1:
                 break
             else:
                 source_node=arr[2]
                 source_index = np.where(nodes == source_node)
                 rates[source_index]=rates[source_index]+float(arr[6])
               
             
       
         for i in range(0,len(nodes)):
             rows.append({'source': i,
                     'bit_rate': rates[i]})      
         
         
     df = pd.DataFrame(rows, columns=cols)
    # Writing dataframe to csv
     df.to_csv('output_abilene.csv')    
